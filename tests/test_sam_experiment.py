"""Guarded SAM experiment tests. All API calls are mocked — no paid calls,
no network, no credentials needed."""

import os
import unittest
from types import SimpleNamespace

from tests import TempDirTest
from reforge.budget import BudgetGuard, BudgetPolicy
from reforge.sam_experiment import run_sam_experiment
from reforge.usage_ledger import UsageLedger, sha256_file


def make_image(case: TempDirTest, name="part.jpg", size=(120, 100)):
    from PIL import Image

    path = case.path(name)
    Image.new("RGB", size, (180, 180, 180)).save(path)
    return path


def fake_records():
    box = SimpleNamespace(
        kind="box", object_id="o1", left=10.0, top=20.0,
        right=100.0, bottom=80.0, confidence=0.9)
    mask = SimpleNamespace(
        kind="mask", object_id="o1",
        bounds=SimpleNamespace(left=10.0, top=20.0, right=100.0, bottom=80.0),
        mask=SimpleNamespace(encoding="one_bit", payload="!!!!", width=4, height=4),
        confidence=0.8)
    return [box, mask]


def fake_result(status="completed"):
    return SimpleNamespace(
        records=fake_records(), diagnostics=[],
        outcome=SimpleNamespace(status=status))


class FakeClient:
    """Mock API client counting exactly how many requests were sent."""

    def __init__(self, result=None, error=None):
        self.calls = 0
        self._result = result if result is not None else fake_result()
        self._error = error
        self.responses = SimpleNamespace(create=self.create)

    async def create(self, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return object()  # events; parse_stream mock ignores them

    class _Parsed:
        def __init__(self, result):
            self._result = result

        async def final_result(self):
            return self._result


def parse_ok(events):
    return FakeClient._Parsed(fake_result())


def fake_decode(mask):
    return bytes([0, 1] * 8)  # 4x4 one-bit raster


def run(case: TempDirTest, image, client, **kwargs):
    kwargs.setdefault("concept", "flat metal plate with circular holes")
    kwargs.setdefault("out_path", case.path("overlay.png"))
    kwargs.setdefault("experiment_id", "test-sam-001")
    kwargs.setdefault("ledger_path", case.path("usage.jsonl"))
    kwargs.setdefault("cache_dir", case.path("cache"))
    kwargs.setdefault("decode_mask", fake_decode)
    kwargs.setdefault("parse_stream", parse_ok)
    factory_calls = []

    def factory():
        factory_calls.append(1)
        return client

    kwargs.setdefault("client_factory", factory)
    result = run_sam_experiment(image_path=image, **kwargs)
    return result, client, factory_calls


class TestGuardEnforcement(TempDirTest):
    def test_blocked_request_sends_nothing(self):
        image = make_image(self)
        client = FakeClient()
        result, client, factory_calls = run(
            self, image, client,
            policy=BudgetPolicy(per_experiment_cap_usd=1.0), max_cost_usd=50.0)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(client.calls, 0)
        self.assertEqual(factory_calls, [])
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].status, "blocked")

    def test_no_automatic_retry_on_failure(self):
        image = make_image(self)
        client = FakeClient(error=RuntimeError("boom"))
        result, client, _ = run(self, image, client)
        self.assertEqual(result.status, "error")
        self.assertEqual(client.calls, 1)  # exactly one attempt, no retry

    def test_missing_credentials_blocked_without_call(self):
        image = make_image(self)
        old = os.environ.pop("MODEL_API_KEY", None)
        self.addCleanup(lambda: os.environ.update({"MODEL_API_KEY": old}) if old else None)
        client = FakeClient()
        result, client, factory_calls = run(
            self, image, client, client_factory=None)  # default factory, no key
        self.assertEqual(result.status, "blocked")
        self.assertEqual(client.calls, 0)
        self.assertEqual(factory_calls, [])

    def test_guard_unavailable_refuses_to_call(self):
        image = make_image(self)
        os.makedirs(self.path("usage.jsonl"))  # ledger path unusable
        client = FakeClient()
        result, client, _ = run(self, image, client)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(client.calls, 0)


class TestLedgerRecording(TempDirTest):
    def test_success_recorded_with_hash_and_counts(self):
        image = make_image(self)
        client = FakeClient()
        result, _, _ = run(self, image, client)
        self.assertEqual(result.status, "ok")
        self.assertTrue(os.path.isfile(self.path("overlay.png")))
        self.assertEqual((result.box_count, result.mask_count), (1, 1))
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.status, "ok")
        self.assertEqual(entry.model, "sam-3.1")
        self.assertEqual(entry.operation, "image_segmentation")
        self.assertEqual(entry.asset_hash, sha256_file(image))
        self.assertEqual(entry.image_count, 1)
        self.assertEqual(entry.cost_status, "unknown")  # provider reports no charge

    def test_failed_response_recorded(self):
        image = make_image(self)
        client = FakeClient(error=RuntimeError("boom"))
        result, _, _ = run(self, image, client)
        self.assertEqual(result.status, "error")
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(entries[0].status, "error")
        self.assertEqual(entries[0].cost_status, "unknown")

    def test_early_outcome_recorded_as_error(self):
        image = make_image(self)

        def parse_early(events):
            return FakeClient._Parsed(fake_result(status="incomplete"))

        client = FakeClient()
        result, _, _ = run(self, image, client, parse_stream=parse_early)
        self.assertEqual(result.status, "error")
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(entries[0].status, "error")

    def test_unknown_cost_not_treated_as_zero(self):
        image = make_image(self)
        client = FakeClient()
        run(self, image, client)
        guard = BudgetGuard(BudgetPolicy(), UsageLedger(self.path("usage.jsonl")))
        self.assertAlmostEqual(guard.spent_usd(), 5.0)  # assumption, never zero

    def test_no_secrets_in_ledger(self):
        os.environ["MODEL_API_KEY"] = "sk-test-SECRET-VALUE-xyz"
        self.addCleanup(lambda: os.environ.pop("MODEL_API_KEY", None))
        image = make_image(self)
        client = FakeClient()
        run(self, image, client)
        with open(self.path("usage.jsonl"), encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotIn("SECRET", raw)
        self.assertNotIn("base64", raw)  # no image payload logged
        for line in raw.strip().split("\n"):
            self.assertNotIn("flat metal plate", line)  # concept not in ledger


class TestCacheReuse(TempDirTest):
    def test_identical_request_reuses_cache_without_call(self):
        image = make_image(self)
        client = FakeClient()
        first, _, _ = run(self, image, client)
        self.assertEqual(first.status, "ok")
        self.assertEqual(client.calls, 1)
        second, _, factory_calls = run(self, image, client)
        self.assertEqual(second.status, "cached")
        self.assertTrue(second.from_cache)
        self.assertEqual(client.calls, 1)  # no second request
        self.assertTrue(os.path.isfile(self.path("overlay.png")))
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(entries[-1].status, "cached")
        guard = BudgetGuard(BudgetPolicy(), UsageLedger(self.path("usage.jsonl")))
        self.assertAlmostEqual(guard.spent_usd(), 5.0)  # cache hit adds nothing

    def test_different_concept_is_not_a_cache_hit(self):
        image = make_image(self)
        client = FakeClient()
        run(self, image, client)
        second, _, _ = run(self, image, client, concept="a duck on a lake")
        self.assertEqual(second.status, "ok")
        self.assertEqual(client.calls, 2)


class TestCliWiring(TempDirTest):
    def test_cli_blocked_exits_3_without_call(self):
        from reforge.cli import main

        image = make_image(self)
        code = main(["sam-segment", "--image", image,
                     "--concept", "flat metal plate",
                     "--out", self.path("overlay.png"),
                     "--ledger", self.path("usage.jsonl"),
                     "--cache-dir", self.path("cache"),
                     "--max-cost", "500.0"])  # exceeds default $5 cap
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
