"""Responses-API contract tests: assembly, strict schema, usage parsing.

Fully offline: local fixtures and mocked transports only. No network, no
paid calls, no credentials needed.
"""

import json
import os
import unittest

from tests import TempDirTest
from reforge.geometry import TOP_LEVEL_KEYS, validate_proposal
from reforge.spark_request import (
    build_responses_body,
    build_spark_request,
    parse_responses_usage,
    parse_spark_response,
    send_spark_request,
)
from reforge.spark_schema import (
    GEOMETRY_PROPOSAL_JSON_SCHEMA,
    RESPONSE_FORMAT_DESCRIPTOR,
    check_strict_compliant,
)

try:
    from test_geometry import valid_proposal_dict
except ImportError:  # run as tests.test_spark_contract instead of discover
    from tests.test_geometry import valid_proposal_dict


def make_image(case: TempDirTest, name="part.jpg", size=(64, 48), fmt="JPEG"):
    from PIL import Image

    path = case.path(name)
    Image.new("RGB", size, (180, 180, 180)).save(path, fmt)
    return path


def make_request(case: TempDirTest, **kwargs):
    image = make_image(case)
    return build_spark_request(image, None, **kwargs), image


def send_kwargs(case: TempDirTest):
    return {"ledger_path": case.path("usage.jsonl"),
            "cache_dir": case.path("cache"),
            "experiment_id": "test-spark-contract"}


class TestRequestAssembly(TempDirTest):
    def test_documented_wire_fields(self):
        request, _ = make_request(self)
        body = build_responses_body(request)
        self.assertEqual(body["model"], "muse-spark-1.3")
        self.assertEqual(len(body["input"]), 1)
        message = body["input"][0]
        self.assertEqual(message["role"], "user")
        kinds = [block["type"] for block in message["content"]]
        self.assertEqual(kinds, ["input_text", "input_image"])
        self.assertEqual(message["content"][0]["text"], request.instructions)

    def test_image_data_url_representation(self):
        request, image = make_request(self)
        body = build_responses_body(request)
        url = body["input"][0]["content"][1]["image_url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        import base64

        payload = url.split(",", 1)[1]
        self.assertGreater(len(base64.b64decode(payload)), 0)

    def test_structured_output_descriptor(self):
        request, _ = make_request(self)
        body = build_responses_body(request)
        format_desc = body["text"]["format"]
        self.assertEqual(format_desc["type"], "json_schema")
        self.assertTrue(format_desc["strict"])
        self.assertEqual(format_desc["schema"], GEOMETRY_PROPOSAL_JSON_SCHEMA)
        self.assertEqual(RESPONSE_FORMAT_DESCRIPTOR["schema"],
                         GEOMETRY_PROPOSAL_JSON_SCHEMA)

    def test_deterministic_construction(self):
        request, _ = make_request(self)
        self.assertEqual(build_responses_body(request),
                         build_responses_body(request))

    def test_no_credentials_or_network_needed(self):
        old = os.environ.pop("MODEL_API_KEY", None)
        self.addCleanup(
            lambda: os.environ.update({"MODEL_API_KEY": old}) if old else None)
        request, _ = make_request(self)
        body = build_responses_body(request)  # must not raise without a key
        self.assertNotIn("MODEL_API_KEY", json.dumps(body))

    def test_unsupported_mime_rejected(self):
        path = make_image(self, name="part.bmp", fmt="BMP")
        request = build_spark_request(path, None)
        with self.assertRaises(Exception) as ctx:
            build_responses_body(request)
        self.assertIn("MIME", str(ctx.exception))


class TestStrictSchema(unittest.TestCase):
    def test_schema_is_strict_compliant(self):
        self.assertEqual(check_strict_compliant(GEOMETRY_PROPOSAL_JSON_SCHEMA), [])

    def test_self_check_rejects_bad_schemas(self):
        self.assertTrue(check_strict_compliant(
            {"type": "object", "properties": {"a": {"type": "string"}}}))
        self.assertTrue(check_strict_compliant(
            {"type": "object", "additionalProperties": False,
             "required": ["a"], "properties": {"a": {"anyOf": []}}}))

    def test_schema_covers_application_contract(self):
        from reforge.geometry import MEASUREMENT_KEYS

        required = set(GEOMETRY_PROPOSAL_JSON_SCHEMA["required"])
        # Every locally required top-level key except server-unknowable
        # evidence must be producible under the provider schema.
        self.assertTrue(set(TOP_LEVEL_KEYS) - {"evidence"} <= required)
        holes = GEOMETRY_PROPOSAL_JSON_SCHEMA["properties"]["holes"]["items"]
        self.assertEqual(set(holes["required"]), {"x", "y", "diameter", "depth"})
        length = GEOMETRY_PROPOSAL_JSON_SCHEMA["properties"]["length"]
        self.assertEqual(set(length["required"]), set(MEASUREMENT_KEYS))
        self.assertEqual(
            GEOMETRY_PROPOSAL_JSON_SCHEMA["properties"]["classification"]["enum"],
            ["rectangular_plate"])

    def test_schema_shaped_output_still_validates_locally(self):
        # The provider schema must not admit anything the local validator
        # would wave through weakened: a valid fixture validates, and the
        # local validator remains the final authority for everything the
        # schema cannot express (ranges, consistency, confirmations).
        self.assertIsNotNone(validate_proposal(valid_proposal_dict()))
        raw = valid_proposal_dict()
        raw["holes"][0]["x"]["value"] = 500.0  # schema-plausible, locally absurd
        from reforge.geometry import ProposalError

        with self.assertRaises(ProposalError):
            validate_proposal(raw)


class TestUsageParsing(unittest.TestCase):
    def test_full_nested_usage(self):
        parsed = parse_responses_usage({
            "input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens_details": {"reasoning_tokens": 50}})
        self.assertEqual(
            (parsed["input_tokens"], parsed["output_tokens"],
             parsed["total_tokens"], parsed["cached_tokens"]),
            (1200, 300, 1500, 200))

    def test_missing_usage_is_unknown(self):
        self.assertEqual(
            parse_responses_usage(None),
            {"input_tokens": None, "output_tokens": None,
             "total_tokens": None, "cached_tokens": None})

    def test_partial_usage(self):
        parsed = parse_responses_usage({"input_tokens": 1200})
        self.assertEqual(parsed["input_tokens"], 1200)
        self.assertIsNone(parsed["output_tokens"])
        self.assertIsNone(parsed["cached_tokens"])

    def test_malformed_usage_never_invented(self):
        for bad in ("nope", [1, 2], 42,
                    {"input_tokens": "many"},
                    {"input_tokens": True},
                    {"input_tokens": -5},
                    {"input_tokens_details": [1]},
                    {"input_tokens_details": {"cached_tokens": "some"}}):
            parsed = parse_responses_usage(bad)
            self.assertTrue(all(v is None or v >= 0 for v in parsed.values()),
                            msg=repr(bad))
            if isinstance(bad, dict) and bad.get("input_tokens") == "many":
                self.assertIsNone(parsed["input_tokens"])


class TestUsageThroughSend(TempDirTest):
    def test_envelope_usage_recorded_with_cached_note(self):
        from reforge.usage_ledger import UsageLedger

        def transport(request):
            return {"output": json.dumps(valid_proposal_dict()),
                    "usage": {"input_tokens": 1200, "output_tokens": 300,
                              "total_tokens": 1500,
                              "input_tokens_details": {"cached_tokens": 200}}}

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "ok")
        entry = UsageLedger(self.path("usage.jsonl")).entries()[0]
        self.assertEqual((entry.input_tokens, entry.output_tokens), (1200, 300))
        self.assertIn("cached_input_tokens=200", entry.note)
        self.assertEqual(entry.cost_status, "unknown")  # tokens are not charges

    def test_bare_output_without_usage_stays_unknown(self):
        from reforge.usage_ledger import UsageLedger

        def transport(request):
            return json.dumps(valid_proposal_dict())

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "ok")
        entry = UsageLedger(self.path("usage.jsonl")).entries()[0]
        self.assertIsNone(entry.input_tokens)
        self.assertEqual(entry.cost_status, "unknown")

    def test_envelope_with_unparseable_output_fails_closed(self):
        def transport(request):
            return {"output": "difference() { cube(1); }",
                    "usage": {"input_tokens": 10}}

        request, _ = make_request(self)
        result = send_spark_request(request, transport=transport, allow_live=True,
                                    **send_kwargs(self))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.proposal)


class TestBodyHygiene(TempDirTest):
    def test_body_never_persisted_to_cache_or_ledger(self):
        from reforge.usage_ledger import UsageLedger

        os.environ["MODEL_API_KEY"] = "sk-test-SECRET-VALUE-xyz"
        self.addCleanup(lambda: os.environ.pop("MODEL_API_KEY", None))

        def transport(request):
            return json.dumps(valid_proposal_dict())

        request, _ = make_request(self)
        build_responses_body(request)  # transient only; returned, not stored
        send_spark_request(request, transport=transport, allow_live=True,
                           **send_kwargs(self))
        with open(self.path("usage.jsonl"), encoding="utf-8") as handle:
            self.assertNotIn("SECRET", handle.read())
            self.assertNotIn("base64", handle.read())
        cache_files = os.listdir(self.path("cache"))
        self.assertEqual(len(cache_files), 1)
        with open(os.path.join(self.path("cache"), cache_files[0]),
                  encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotIn("SECRET", raw)
        self.assertNotIn("base64", raw)  # no image bytes in cache metadata


if __name__ == "__main__":
    unittest.main()
