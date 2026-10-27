"""Evidence-to-request dry-run tests. Fully offline: fixtures and injected
decoders only. No network, no paid calls, no credentials needed."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from tests import TempDirTest
from reforge.budget import BudgetPolicy
from reforge.probe import ProbeError, prepare_probe
from reforge.spark_request import SPARK_CACHE_VERSION, SPARK_MODEL_DEFAULT
from reforge.usage_ledger import sha256_file


def make_image(case: TempDirTest, name="part.jpg", size=(64, 48)):
    from PIL import Image

    path = case.path(name)
    Image.new("RGB", size, (180, 180, 180)).save(path)
    return path


def mask_summary_for(case: TempDirTest, raster, image_hash, name="mask.json"):
    height, width = raster.shape
    summary = {
        "v": 1, "model": "sam-3.1", "concept": "flat metal plate",
        "asset_hash": image_hash, "boxes": [],
        "masks": [{
            "kind": "mask", "object_id": "o1",
            "bounds": {"left": 5.0, "top": 5.0,
                       "right": 5.0 + width, "bottom": 5.0 + height},
            "mask": {"encoding": "one_bit", "payload": "<fixture>",
                     "width": width, "height": height},
            "confidence": 0.9}],
        "diagnostics": [], "outcome": "completed",
    }
    path = case.path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle)
    return path


def fake_decode(raster):
    blob = raster.astype(np.uint8).tobytes()

    def decode(mask):
        return blob

    return decode


def prepare(case: TempDirTest, image, summary=None, **kwargs):
    kwargs.setdefault("ledger_path", case.path("usage.jsonl"))
    return prepare_probe(image, summary, **kwargs)


class TestValidDryRun(TempDirTest):
    def test_fixture_produces_request_and_report(self):
        image = make_image(self)
        raster = np.ones((20, 30), dtype=bool)
        summary = mask_summary_for(self, raster, sha256_file(image))
        report = prepare(self, image, summary, decode_fn=fake_decode(raster))
        self.assertFalse(report["transmitted"])
        self.assertEqual(report["image"]["sha256"], sha256_file(image))
        self.assertEqual((report["image"]["width_px"], report["image"]["height_px"]),
                         (64, 48))
        self.assertTrue(report["evidence"]["present"])
        self.assertTrue(report["evidence"]["asset_hash_match"])
        self.assertEqual(report["request"]["model"], SPARK_MODEL_DEFAULT)
        self.assertEqual(report["request"]["classification"], "rectangular_plate")
        self.assertEqual(report["request"]["cache_version"], SPARK_CACHE_VERSION)
        self.assertIn(report["guard"]["verdict"], ("would-allow", "would-block"))
        self.assertIn("disabled", report["transport"])

    def test_missing_summary_is_explicit_gap(self):
        image = make_image(self)
        report = prepare(self, image, None)
        self.assertFalse(report["transmitted"])
        self.assertFalse(report["evidence"]["present"])
        self.assertIn("gap", report["evidence"])

    def test_report_is_deterministic(self):
        image = make_image(self)
        first = prepare(self, image, None)
        second = prepare(self, image, None)
        self.assertEqual(first, second)
        json.dumps(first)  # serializable: no bytes objects

    def test_no_base64_or_bytes_in_report(self):
        image = make_image(self)
        raster = np.ones((20, 30), dtype=bool)
        summary = mask_summary_for(self, raster, sha256_file(image))
        report = prepare(self, image, summary, decode_fn=fake_decode(raster))
        self.assertNotIn("base64", json.dumps(report))

    def test_guard_is_invoked(self):
        image = make_image(self)
        allowed = prepare(self, image, None)
        self.assertEqual(allowed["guard"]["verdict"], "would-allow")
        blocked = prepare(
            self, image, None, max_cost_usd=50.0,
            policy=BudgetPolicy(per_experiment_cap_usd=1.0))
        self.assertEqual(blocked["guard"]["verdict"], "would-block")


class TestSafeFailures(TempDirTest):
    def test_invalid_image_fails_safely(self):
        path = self.path("not-an-image.jpg")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("this is text, not pixels")
        with self.assertRaises(ProbeError):
            prepare(self, path, None)

    def test_missing_image_fails_safely(self):
        with self.assertRaises(ProbeError):
            prepare(self, self.path("absent.jpg"), None)

    def test_malformed_summary_fails_safely(self):
        image = make_image(self)
        path = self.path("bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"v": 99, "masks": []}')
        with self.assertRaises(ProbeError):
            prepare(self, image, path)

    def test_hash_mismatch_is_hard_error(self):
        image = make_image(self)
        raster = np.ones((20, 30), dtype=bool)
        summary = mask_summary_for(self, raster, "00" * 32)
        with self.assertRaises(ProbeError) as ctx:
            prepare(self, image, summary, decode_fn=fake_decode(raster))
        self.assertIn("does not match", str(ctx.exception))


class TestNoTransportNoCredentials(TempDirTest):
    def test_no_network_reachable_from_command(self):
        image = make_image(self)
        with mock.patch("socket.create_connection",
                         side_effect=AssertionError("network used")):
            with mock.patch("socket.socket",
                             side_effect=AssertionError("network used")):
                report = prepare(self, image, None)
        self.assertFalse(report["transmitted"])

    def test_no_credentials_required(self):
        old = os.environ.pop("MODEL_API_KEY", None)
        self.addCleanup(
            lambda: os.environ.update({"MODEL_API_KEY": old}) if old else None)
        image = make_image(self)
        report = prepare(self, image, None)
        self.assertFalse(report["transmitted"])

    def test_ledger_and_cache_untouched(self):
        image = make_image(self)
        ledger_path = self.path("usage.jsonl")
        prepare(self, image, None)
        self.assertFalse(os.path.exists(ledger_path))

    def test_cli_reports_not_transmitted(self):
        from reforge.cli import main

        image = make_image(self)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["probe-dry-run", "--image", image,
                         "--ledger", self.path("usage.jsonl")])
        self.assertEqual(code, 0)
        report = json.loads(buffer.getvalue())
        self.assertFalse(report["transmitted"])
        self.assertIn("no request was sent", report["note"])

    def test_cli_bad_image_exits_2(self):
        from reforge.cli import main

        code = main(["probe-dry-run", "--image", self.path("absent.jpg")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
