"""Spark request-preparation tests. Fully offline: mocked transports only,
no network, no paid calls, no credentials needed."""

import json
import os
import unittest

from tests import TempDirTest
from reforge.budget import BudgetPolicy
from reforge.geometry import ProposalError
from reforge.spark_request import (
    LiveCallsDisabledError,
    SparkRequestError,
    build_spark_request,
    dry_run,
    parse_spark_response,
    send_spark_request,
)
from reforge.usage_ledger import UsageLedger, sha256_file

try:
    from test_geometry import valid_proposal_dict
except ImportError:  # run as tests.test_spark_request instead of discover
    from tests.test_geometry import valid_proposal_dict


def make_image(case: TempDirTest, name="part.jpg", size=(64, 48)):
    from PIL import Image

    path = case.path(name)
    Image.new("RGB", size, (180, 180, 180)).save(path)
    return path


EVIDENCE = {"bounds_px": (10.0, 10.0, 50.0, 30.0), "fill_fraction": 0.95,
            "interior_gap_count": 0, "box_confidence": 0.9}


def make_request(case: TempDirTest, **kwargs):
    image = make_image(case)
    kwargs.setdefault("mask_evidence", EVIDENCE)
    return build_spark_request(image, **kwargs), image


def send_kwargs(case: TempDirTest):
    return {"ledger_path": case.path("usage.jsonl"),
            "cache_dir": case.path("cache"),
            "experiment_id": "test-spark-001"}


class TestRequestConstruction(TempDirTest):
    def test_build_fields(self):
        request, image = make_request(self)
        self.assertEqual(request.model, "muse-spark-1.3")
        self.assertEqual(request.image_asset_hash, sha256_file(image))
        self.assertEqual(request.image_size_px, (64, 48))
        self.assertEqual(request.classification, "rectangular_plate")
        self.assertIn("rectangular_plate", request.instructions)
        self.assertIn("plate corner", request.instructions)

    def test_image_space_not_treated_as_mm(self):
        request, _ = make_request(self)
        self.assertIn("PIXELS, not millimetres", request.instructions)
        self.assertIn("never convert", request.instructions)
        self.assertNotIn("px/mm", request.instructions)

    def test_unknowns_must_stay_unknown(self):
        request, _ = make_request(self)
        self.assertIn("stay null", request.instructions)
        self.assertIn("never fill them with plausible guesses", request.instructions)
        self.assertIn("requires_confirmation", request.instructions)

    def test_no_executable_output_allowed(self):
        request, _ = make_request(self)
        self.assertIn("Forbidden", request.instructions)
        self.assertIn("JSON data only", request.instructions)

    def test_unsupported_classification_rejected(self):
        image = make_image(self)
        with self.assertRaises(SparkRequestError):
            build_spark_request(image, EVIDENCE, classification="washer")

    def test_missing_image_rejected(self):
        with self.assertRaises(SparkRequestError):
            build_spark_request(self.path("nope.jpg"), EVIDENCE)

    def test_no_evidence_still_builds(self):
        image = make_image(self)
        request = build_spark_request(image, None)
        self.assertIn("unknown", request.instructions)


class TestResponseParsing(unittest.TestCase):
    def test_valid_json_parses(self):
        proposal = parse_spark_response(json.dumps(valid_proposal_dict()))
        self.assertEqual(len(proposal.holes), 4)

    def test_fenced_json_parses(self):
        text = "```json\n" + json.dumps(valid_proposal_dict()) + "\n```"
        self.assertIsNotNone(parse_spark_response(text))

    def test_malformed_json_rejected(self):
        with self.assertRaises(ProposalError):
            parse_spark_response('{"classification": ')

    def test_invalid_proposal_rejected(self):
        raw = valid_proposal_dict()
        raw["classification"] = "washer"
        with self.assertRaises(ProposalError):
            parse_spark_response(json.dumps(raw))

    def test_cad_code_rejected(self):
        with self.assertRaises(ProposalError) as ctx:
            parse_spark_response("difference() { cube([60, 40, 5]); }")
        self.assertIn("executable", str(ctx.exception))

    def test_mapping_with_cube_rejected(self):
        raw = valid_proposal_dict()
        raw["assumptions"] = ["render with cube([58, 38, 5]) for preview"]
        with self.assertRaises(ProposalError) as ctx:
            parse_spark_response(raw)
        self.assertIn("assumptions", str(ctx.exception))

    def test_mapping_with_difference_rejected(self):
        raw = valid_proposal_dict()
        raw["unknowns"] = ["whether to difference() the corners"]
        with self.assertRaises(ProposalError):
            parse_spark_response(raw)

    def test_mapping_and_string_screened_consistently(self):
        # Equivalent content as a mapping and as JSON text: both rejected.
        raw = valid_proposal_dict()
        raw["assumptions"] = ["preview via cylinder(h=7, d=6)"]
        with self.assertRaises(ProposalError):
            parse_spark_response(raw)
        with self.assertRaises(ProposalError):
            parse_spark_response(json.dumps(raw))

    def test_legitimate_mapping_accepted(self):
        proposal = parse_spark_response(valid_proposal_dict())
        self.assertEqual(len(proposal.holes), 4)

    def test_oversized_response_rejected(self):
        from reforge.spark_request import MAX_RESPONSE_CHARS

        with self.assertRaises(ProposalError):
            parse_spark_response("x" * (MAX_RESPONSE_CHARS + 1))

    def test_non_text_non_mapping_rejected(self):
        with self.assertRaises(ProposalError):
            parse_spark_response(42)


class TestDryRun(TempDirTest):
    def test_dry_run_makes_no_call_and_writes_nothing(self):
        request, _ = make_request(self)
        ledger_path = self.path("usage.jsonl")
        report = dry_run(request, ledger_path=ledger_path)
        self.assertEqual(report["verdict"], "would-allow")
        self.assertEqual(report["action"], "dry-run (no request sent)")
        self.assertFalse(os.path.exists(ledger_path))  # read-only: nothing written
        self.assertIn("unconfigured", report["transport"])

    def test_dry_run_reports_block(self):
        request, _ = make_request(self, max_cost_usd=50.0)
        report = dry_run(request, policy=BudgetPolicy(per_experiment_cap_usd=1.0),
                         ledger_path=self.path("usage.jsonl"))
        self.assertEqual(report["verdict"], "would-block")
        self.assertIn("cap", report["reason"])


class TestSendDiscipline(TempDirTest):
    def test_disabled_by_default(self):
        request, _ = make_request(self)
        with self.assertRaises(LiveCallsDisabledError):
            send_spark_request(request, **send_kwargs(self))
        with self.assertRaises(LiveCallsDisabledError):
            send_spark_request(request, transport=lambda r: "{}",
                               allow_live=False, **send_kwargs(self))
        self.assertFalse(os.path.exists(self.path("usage.jsonl")))

    def test_budget_preflight_blocks_before_transmission(self):
        calls = []

        def transport(request):
            calls.append(request)
            return "{}"

        request, _ = make_request(self, max_cost_usd=50.0)
        result = send_spark_request(
            request, transport=transport, allow_live=True,
            policy=BudgetPolicy(per_experiment_cap_usd=1.0),
            **send_kwargs(self))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(calls, [])  # transport never invoked
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(entries[0].status, "blocked")

    def test_mock_success_recorded_and_cached(self):
        calls = []

        def transport(request):
            calls.append(request)
            return json.dumps(valid_proposal_dict())

        request, _ = make_request(self)
        first = send_spark_request(request, transport=transport, allow_live=True,
                                   **send_kwargs(self))
        self.assertEqual(first.status, "ok")
        self.assertEqual(len(first.proposal.holes), 4)
        second = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(second.status, "cached")
        self.assertTrue(second.from_cache)
        self.assertEqual(len(calls), 1)  # no second transmission
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(entries[0].status, "ok")
        self.assertEqual(entries[0].cost_status, "unknown")  # pricing unverified
        self.assertEqual(entries[-1].status, "cached")

    def test_transport_error_recorded_without_retry(self):
        calls = []

        def transport(request):
            calls.append(request)
            raise ConnectionError("network down")

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "error")
        self.assertEqual(len(calls), 1)  # exactly one attempt, no retry
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(entries[0].status, "error")
        self.assertEqual(entries[0].cost_status, "unknown")

    def test_malformed_response_recorded(self):
        def transport(request):
            return "not json at all {{{"

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.proposal)

    def test_cache_version_is_deterministic(self):
        import reforge.spark_request as spark_module

        request, _ = make_request(self)
        self.assertEqual(spark_module._cache_key(request),
                         spark_module._cache_key(request))
        self.assertIn("request-template-v1", spark_module._cache_key(request))

    def test_changing_version_changes_key(self):
        import reforge.spark_request as spark_module

        request, _ = make_request(self)
        before = spark_module._cache_key(request)
        old = spark_module.SPARK_CACHE_VERSION
        spark_module.SPARK_CACHE_VERSION = old + 1
        try:
            self.assertNotEqual(spark_module._cache_key(request), before)
        finally:
            spark_module.SPARK_CACHE_VERSION = old

    def test_old_version_entry_is_not_reused(self):
        import reforge.spark_request as spark_module
        from reforge.budget import BudgetPolicy
        from reforge.budget import ResultCache

        request, _ = make_request(self, max_cost_usd=50.0)
        calls = []

        def transport(r):
            calls.append(r)
            return json.dumps(valid_proposal_dict())

        # Plant a valid entry under a previous version identity.
        old = spark_module.SPARK_CACHE_VERSION
        spark_module.SPARK_CACHE_VERSION = old - 1
        try:
            old_key = spark_module._cache_key(request)
            ResultCache(self.path("cache")).put(
                old_key, {"v": 1, "request_version": old - 1,
                          "model": request.model,
                          "asset_hash": request.image_asset_hash,
                          "response": json.dumps(valid_proposal_dict())})
        finally:
            spark_module.SPARK_CACHE_VERSION = old
        # Current version must miss and reach the (denying) guard, not a hit.
        result = send_spark_request(
            request, transport=transport, allow_live=True,
            policy=BudgetPolicy(per_experiment_cap_usd=1.0),
            **send_kwargs(self))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(calls, [])

    def test_version_mismatch_in_current_key_is_not_reused(self):
        import reforge.spark_request as spark_module
        from reforge.budget import BudgetPolicy
        from reforge.budget import ResultCache

        request, _ = make_request(self, max_cost_usd=50.0)
        key = spark_module._cache_key(request)
        ResultCache(self.path("cache")).put(
            key, {"v": 1, "request_version": -1, "model": request.model,
                  "asset_hash": request.image_asset_hash,
                  "response": json.dumps(valid_proposal_dict())})
        calls = []
        result = send_spark_request(
            request, transport=lambda r: calls.append(r) or "{}",
            allow_live=True, policy=BudgetPolicy(per_experiment_cap_usd=1.0),
            **send_kwargs(self))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(calls, [])

    def test_stored_entry_carries_version(self):
        import reforge.spark_request as spark_module

        def transport(request):
            return json.dumps(valid_proposal_dict())

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "ok")
        key = spark_module._cache_key(request)
        entry = spark_module.ResultCache(self.path("cache")).get(key)
        self.assertEqual(entry["request_version"], spark_module.SPARK_CACHE_VERSION)

    def test_transport_receives_wire_body_not_request_object(self):
        import json as _json

        seen = []

        def transport(body):
            seen.append(body)
            return _json.dumps(valid_proposal_dict())

        from reforge.spark_request import SparkRequest

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(seen), 1)
        body = seen[0]
        self.assertNotIsInstance(body, SparkRequest)
        self.assertEqual(body["model"], request.model)
        self.assertEqual(body["input"][0]["content"][0]["text"],
                         request.instructions)
        self.assertTrue(body["input"][0]["content"][1]["image_url"].startswith(
            "data:image/jpeg;base64,"))

    def test_guard_runs_before_wire_assembly(self):
        from reforge.budget import BudgetPolicy
        from reforge.spark_request import SparkRequest

        calls = []
        # Request points at a missing image: wire assembly would fail, but the
        # denying guard must win first with a blocked verdict, not an error.
        request = SparkRequest(
            model="muse-spark-1.3", image_path=self.path("absent.jpg"),
            image_asset_hash="00" * 32, image_size_px=(64, 48),
            concept="flat metal plate", classification="rectangular_plate",
            mask_bounds_px=None, mask_fill_fraction=None, interior_gap_count=None,
            box_confidence=None, instructions="x", max_cost_usd=50.0)
        result = send_spark_request(
            request, transport=lambda body: calls.append(body),
            allow_live=True, policy=BudgetPolicy(per_experiment_cap_usd=1.0),
            **send_kwargs(self))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(calls, [])

    def test_envelope_missing_output_fails_closed(self):
        def transport(body):
            return {"usage": {"input_tokens": 10}}  # no model output at all

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.proposal)

    def test_no_secrets_in_ledger_or_cache(self):
        os.environ["MODEL_API_KEY"] = "sk-test-SECRET-VALUE-xyz"
        self.addCleanup(lambda: os.environ.pop("MODEL_API_KEY", None))

        def transport(request):
            return json.dumps(valid_proposal_dict())

        request, _ = make_request(self)
        dry_run(request, ledger_path=self.path("usage.jsonl"))
        send_spark_request(request, transport=transport, allow_live=True,
                           **send_kwargs(self))
        with open(self.path("usage.jsonl"), encoding="utf-8") as handle:
            ledger_raw = handle.read()
        self.assertNotIn("SECRET", ledger_raw)
        cache_file = os.path.join(self.path("cache"),
                                  os.listdir(self.path("cache"))[0])
        with open(cache_file, encoding="utf-8") as handle:
            self.assertNotIn("SECRET", handle.read())


if __name__ == "__main__":
    unittest.main()
