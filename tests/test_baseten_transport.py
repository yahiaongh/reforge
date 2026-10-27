"""Baseten Chat Completions transport tests.

Fully offline: every network interaction goes through an injected fake
request function; no credentials are real (synthetic keys only, and the
real environment is never read); nothing is transmitted anywhere.
"""

import copy
import json
import os
import socket
import unittest
import urllib.error

from tests import TempDirTest

try:
    from test_geometry import valid_proposal_dict
except ImportError:  # run as tests.test_baseten_transport instead of discover
    from tests.test_geometry import valid_proposal_dict

from reforge.baseten_transport import (
    BASETEN_CHAT_PATH,
    BASETEN_ENV_VAR,
    BASETEN_MAX_TOKENS_DEFAULT,
    BASETEN_MODEL_DEFAULT,
    BasetenAuthorization,
    BasetenChatTransport,
    body_has_image,
    build_baseten_body,
    parse_baseten_usage,
    send_baseten_request,
)
from reforge.probe import prepare_baseten_probe
from reforge.sam_experiment import MissingCredentialsError
from reforge.spark_request import (
    LiveCallsDisabledError,
    SparkRequestError,
    SparkTransportError,
)
from reforge.usage_ledger import UsageLedger

FAKE_KEY = "bt-synthetic-test-key-0001"
OTHER_KEY = "model-synthetic-test-key-0002"


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fake_ok(body_obj, usage=None, finish_reason="stop", record=None):
    def _fn(request, timeout=None):
        if record is not None:
            record.append({"request": request, "timeout": timeout})
        envelope = {"choices": [{"message": {"content": json.dumps(body_obj)},
                                 "finish_reason": finish_reason}]}
        if usage is not None:
            envelope["usage"] = usage
        return FakeHTTPResponse(json.dumps(envelope).encode("utf-8"))
    return _fn


def make_image(case: TempDirTest, name="bpart.jpg", size=(64, 48)):
    from PIL import Image

    path = case.path(name)
    Image.new("RGB", size, (170, 170, 170)).save(path)
    return path


def live_auth(**kwargs):
    kwargs.setdefault("allow_live", True)
    kwargs.setdefault("max_bound_usd", 5.00)
    return BasetenAuthorization(**kwargs)


def send_defaults(case: TempDirTest, **kwargs):
    kwargs.setdefault("ledger_path", case.path("usage.jsonl"))
    kwargs.setdefault("experiment_id", "test-baseten-001")
    return kwargs


class EnvGuardTest(unittest.TestCase):
    """Save/restore the environment so synthetic keys never leak out and
    real credentials are never read."""

    def setUp(self):
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)


class TestBody(TempDirTest):
    def test_defaults(self):
        body = build_baseten_body(instructions="Describe the plate.")
        self.assertEqual(body["model"], BASETEN_MODEL_DEFAULT)
        self.assertEqual(body["messages"],
                         [{"role": "user",
                           "content": [{"type": "text",
                                        "text": "Describe the plate."}]}])
        self.assertEqual(body["max_tokens"], BASETEN_MAX_TOKENS_DEFAULT)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_deterministic(self):
        kwargs = {"instructions": "abc", "max_tokens": 700}
        self.assertEqual(build_baseten_body(**kwargs),
                         build_baseten_body(**kwargs))

    def test_data_url_accepted_but_flagged(self):
        body = build_baseten_body(
            instructions="x", image_data_url="data:image/jpeg;base64,AAAA")
        self.assertTrue(body_has_image(body))
        part = body["messages"][0]["content"][1]
        self.assertEqual(part["type"], "image_url")

    def test_https_url_accepted(self):
        body = build_baseten_body(
            instructions="x", image_data_url="https://example.invalid/p.jpg")
        self.assertTrue(body_has_image(body))

    def test_plain_path_rejected(self):
        with self.assertRaises(SparkRequestError):
            build_baseten_body(instructions="x",
                               image_data_url="/tmp/local.jpg")

    def test_bad_max_tokens(self):
        for bad in (0, -5, True, "1500", 1.5, None):
            with self.assertRaises(SparkRequestError):
                build_baseten_body(instructions="x", max_tokens=bad)

    def test_bad_model_and_instructions(self):
        with self.assertRaises(SparkRequestError):
            build_baseten_body(instructions="x", model="")
        with self.assertRaises(SparkRequestError):
            build_baseten_body(instructions="   ")
        with self.assertRaises(SparkRequestError):
            build_baseten_body(instructions="x",
                               response_format={"no-type": True})

    def test_body_has_image_false(self):
        body = build_baseten_body(instructions="x")
        self.assertFalse(body_has_image(body))
        self.assertFalse(body_has_image({}))
        self.assertFalse(body_has_image({"messages": []}))


class TestTransportConfig(EnvGuardTest):
    def test_exact_default_endpoint(self):
        transport = BasetenChatTransport(api_key=FAKE_KEY,
                                         request_fn=fake_ok({}))
        self.assertEqual(
            transport.url,
            "https://inference.baseten.co/v1" + BASETEN_CHAT_PATH)

    def test_repr_hides_key(self):
        transport = BasetenChatTransport(api_key=FAKE_KEY,
                                         request_fn=fake_ok({}))
        self.assertNotIn(FAKE_KEY, repr(transport))
        self.assertIn("inference.baseten.co", repr(transport))

    def test_rejects_bad_construction(self):
        with self.assertRaises(MissingCredentialsError):
            BasetenChatTransport(api_key="", request_fn=fake_ok({}))
        with self.assertRaises(ValueError):
            BasetenChatTransport(api_key=FAKE_KEY, base_url="http://x",
                                 request_fn=fake_ok({}))
        with self.assertRaises(ValueError):
            BasetenChatTransport(api_key=FAKE_KEY, timeout_s=0,
                                 request_fn=fake_ok({}))

    def test_from_env_uses_baseten_key_only(self):
        os.environ[BASETEN_ENV_VAR] = FAKE_KEY
        os.environ["MODEL_API_KEY"] = OTHER_KEY
        record = []
        transport = BasetenChatTransport.from_env(request_fn=fake_ok(
            {}, record=record))
        transport(build_baseten_body(instructions="x"))
        sent = record[0]["request"]
        self.assertIn(f"Bearer {FAKE_KEY}",
                      sent.get_header("Authorization"))
        self.assertNotIn(OTHER_KEY, sent.get_header("Authorization"))

    def test_from_env_no_fallback_to_model_key(self):
        os.environ.pop(BASETEN_ENV_VAR, None)
        os.environ["MODEL_API_KEY"] = OTHER_KEY
        with self.assertRaises(MissingCredentialsError):
            BasetenChatTransport.from_env(request_fn=fake_ok({}))


class TestWireBehavior(TempDirTest):
    def _transport(self, fn):
        return BasetenChatTransport(api_key=FAKE_KEY, request_fn=fn)

    def test_posts_json_with_bearer_header(self):
        record = []
        transport = self._transport(fake_ok({"ok": True}, record=record))
        body = build_baseten_body(instructions="Describe the plate.",
                                  max_tokens=700)
        transport(body)
        sent = record[0]["request"]
        self.assertEqual(sent.full_url,
                         "https://inference.baseten.co/v1/chat/completions")
        self.assertEqual(sent.get_header("Authorization"), f"Bearer {FAKE_KEY}")
        self.assertNotIn(FAKE_KEY, sent.get_header("Content-type"))
        payload = json.loads(sent.data.decode("utf-8"))
        self.assertEqual(payload["model"], BASETEN_MODEL_DEFAULT)
        self.assertEqual(payload["max_tokens"], 700)
        self.assertEqual(record[0]["timeout"], 120.0)

    def test_http_error_hides_key_and_body(self):
        def _fail(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 429,
                                         "Too Many Requests", {}, None)
        with self.assertRaises(SparkTransportError) as ctx:
            self._transport(_fail)(build_baseten_body(instructions="x"))
        self.assertIn("429", str(ctx.exception))
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_timeout_hides_key(self):
        def _fail(request, timeout=None):
            raise socket.timeout("timed out")
        with self.assertRaises(SparkTransportError) as ctx:
            self._transport(_fail)(build_baseten_body(instructions="x"))
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_malformed_json_envelope(self):
        bad = FakeHTTPResponse(b"not json{{")
        transport = self._transport(lambda request, timeout=None: bad)
        with self.assertRaises(SparkTransportError):
            transport(build_baseten_body(instructions="x"))

    def test_non_mapping_envelope(self):
        bad = FakeHTTPResponse(b"[1, 2]")
        transport = self._transport(lambda request, timeout=None: bad)
        with self.assertRaises(SparkTransportError):
            transport(build_baseten_body(instructions="x"))

    def test_missing_choices_and_content(self):
        for envelope in ({"no": "choices"},
                         {"choices": []},
                         {"choices": [{"message": {}}]},
                         {"choices": [{"message": {"content": None}}]},
                         {"choices": [{"message": {"content": 42}}]}):
            raw = FakeHTTPResponse(json.dumps(envelope).encode("utf-8"))
            transport = self._transport(lambda request, timeout=None: raw)
            with self.assertRaises(SparkTransportError):
                transport(build_baseten_body(instructions="x"))

    def test_list_content_parts_joined(self):
        parts = [{"type": "text", "text": '{"a": 1'},
                 {"type": "image_url", "image_url": {"url": "x"}},
                 {"type": "text", "text": "}"}]
        message = {"content": parts}
        choice = {"message": message, "finish_reason": "stop"}
        envelope = {"choices": [choice]}
        raw = FakeHTTPResponse(json.dumps(envelope).encode("utf-8"))
        transport = self._transport(lambda request, timeout=None: raw)
        out = transport(build_baseten_body(instructions="x"))
        self.assertEqual(out["output"], '{"a": 1\n}')
        self.assertEqual(out["finish_reason"], "stop")

    def test_usage_passes_through(self):
        usage = {"prompt_tokens": 1200, "completion_tokens": 300,
                 "total_tokens": 1500}
        transport = self._transport(fake_ok({"a": 1}, usage=usage))
        out = transport(build_baseten_body(instructions="x"))
        self.assertEqual(out["usage"], usage)


class TestUsageParsing(unittest.TestCase):
    def test_valid_counts(self):
        self.assertEqual(
            parse_baseten_usage({"prompt_tokens": 1200,
                                 "completion_tokens": 300,
                                 "total_tokens": 1500}),
            {"input_tokens": 1200, "output_tokens": 300,
             "total_tokens": 1500})

    def test_partial_counts_keep_missing_as_none(self):
        self.assertEqual(
            parse_baseten_usage({"prompt_tokens": 50}),
            {"input_tokens": 50, "output_tokens": None,
             "total_tokens": None})

    def test_zero_is_not_missing(self):
        self.assertEqual(
            parse_baseten_usage({"prompt_tokens": 0, "completion_tokens": 0,
                                 "total_tokens": 0}),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    def test_missing_and_malformed_are_none(self):
        for bad in (None, {}, {"other": 1}, [1], "x",
                    {"prompt_tokens": "lots"},
                    {"prompt_tokens": -1},
                    {"prompt_tokens": True}):
            self.assertIsNone(parse_baseten_usage(bad), msg=repr(bad))


class TestSendGate(TempDirTest):
    def _calls(self, body_obj=None, **kwargs):
        record = []
        body_obj = body_obj if body_obj is not None else valid_proposal_dict()
        fn = fake_ok(body_obj, record=record, **kwargs)
        transport = BasetenChatTransport(api_key=FAKE_KEY, request_fn=fn)
        return transport, record

    def test_disabled_by_default(self):
        _, record = self._calls()
        body = build_baseten_body(instructions="x")
        with self.assertRaises(LiveCallsDisabledError):
            send_baseten_request(body, transport=None,
                                 auth=live_auth(),
                                 **send_defaults(self))
        self.assertEqual(record, [])

    def test_allow_live_false_denies(self):
        transport, record = self._calls()
        body = build_baseten_body(instructions="x")
        with self.assertRaises(LiveCallsDisabledError):
            send_baseten_request(
                body, transport=transport,
                auth=BasetenAuthorization(allow_live=False,
                                          max_bound_usd=5.00),
                **send_defaults(self))
        self.assertEqual(record, [])

    def test_missing_budget_authorization_blocks(self):
        transport, record = self._calls()
        body = build_baseten_body(instructions="x")
        result = send_baseten_request(
            body, transport=transport, auth=BasetenAuthorization(allow_live=True),
            **send_defaults(self))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(record, [])

    def test_cap_exceeded_blocks(self):
        from reforge.budget import BudgetPolicy
        transport, record = self._calls()
        body = build_baseten_body(instructions="x")
        result = send_baseten_request(
            body, transport=transport, auth=live_auth(max_bound_usd=5.00),
            policy=BudgetPolicy(per_experiment_cap_usd=1.0),
            **send_defaults(self))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(record, [])

    def test_unapproved_image_blocks(self):
        transport, record = self._calls()
        body = build_baseten_body(
            instructions="x", image_data_url="data:image/jpeg;base64,AAAA")
        result = send_baseten_request(
            body, transport=transport, auth=live_auth(),
            **send_defaults(self))
        self.assertEqual(result.status, "blocked")
        self.assertIn("image delivery", result.detail)
        self.assertEqual(record, [])

    def test_approved_image_transmits_once(self):
        transport, record = self._calls()
        body = build_baseten_body(
            instructions="x", image_data_url="data:image/jpeg;base64,AAAA")
        result = send_baseten_request(
            body, transport=transport,
            auth=live_auth(image_delivery_approved=True),
            **send_defaults(self))
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(record), 1)

    def test_invalid_model_or_cap_block(self):
        transport, record = self._calls()
        bad_model = dict(build_baseten_body(instructions="x"))
        bad_model["model"] = ""
        result = send_baseten_request(
            bad_model, transport=transport, auth=live_auth(),
            **send_defaults(self))
        self.assertEqual(result.status, "blocked")
        bad_cap = dict(build_baseten_body(instructions="x"))
        bad_cap["max_tokens"] = 0
        result = send_baseten_request(
            bad_cap, transport=transport, auth=live_auth(),
            **send_defaults(self))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(record, [])

    def test_no_retry_after_failure(self):
        calls = []

        def _flaky(request, timeout=None):
            calls.append(request)
            raise socket.timeout("boom")
        transport = BasetenChatTransport(api_key=FAKE_KEY, request_fn=_flaky)
        result = send_baseten_request(
            build_baseten_body(instructions="x"), transport=transport,
            auth=live_auth(), **send_defaults(self))
        self.assertEqual(result.status, "error")
        self.assertEqual(len(calls), 1)


class TestSendOutcomes(TempDirTest):
    def _send(self, body_obj, **kwargs):
        usage = kwargs.pop("usage", None)
        finish = kwargs.pop("finish_reason", "stop")
        record = []
        transport = BasetenChatTransport(
            api_key=FAKE_KEY,
            request_fn=fake_ok(body_obj, usage=usage,
                               finish_reason=finish, record=record))
        body = build_baseten_body(instructions="x")
        result = send_baseten_request(
            body, transport=transport, auth=live_auth(),
            **send_defaults(self))
        return result, record

    def test_valid_mocked_proposal_validates(self):
        result, record = self._send(
            valid_proposal_dict(),
            usage={"prompt_tokens": 1200, "completion_tokens": 300,
                   "total_tokens": 1500})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.proposal.classification, "rectangular_plate")
        self.assertEqual(result.usage,
                         {"input_tokens": 1200, "output_tokens": 300,
                          "total_tokens": 1500})
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(len(record), 1)
        entries = UsageLedger(self.path("usage.jsonl")).entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].provider, "baseten")
        self.assertEqual(entries[0].status, "ok")
        self.assertEqual(entries[0].cost_status, "unknown")
        self.assertIsNone(entries[0].charge_usd)

    def test_invalid_proposal_rejected(self):
        bad = valid_proposal_dict()
        del bad["length"]
        result, _ = self._send(bad)
        self.assertEqual(result.status, "error")
        self.assertIn("rejected", result.detail)

    def test_cad_marker_rejected(self):
        # A string payload goes through the CAD-marker pre-scan.
        result, _ = self._send('{"note": "cube(10); difference(){}"}')
        self.assertEqual(result.status, "error")
        self.assertIn("rejected", result.detail)

    def test_truncated_output_fails_closed(self):
        result, _ = self._send(valid_proposal_dict(),
                               finish_reason="length")
        self.assertEqual(result.status, "error")
        self.assertIn("truncated", result.detail)
        self.assertIsNone(result.proposal)

    def test_missing_usage_still_ok_with_none(self):
        result, _ = self._send(valid_proposal_dict())
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.usage)

    def test_malformed_usage_still_ok_with_none(self):
        result, _ = self._send(valid_proposal_dict(),
                               usage={"prompt_tokens": "many"})
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.usage)

    def test_unusable_transport_output(self):
        record = []
        transport = BasetenChatTransport(
            api_key=FAKE_KEY,
            request_fn=lambda request, timeout=None: {"no": "output"})
        result = send_baseten_request(
            build_baseten_body(instructions="x"), transport=transport,
            auth=live_auth(), **send_defaults(self))
        self.assertEqual(result.status, "error")
        self.assertEqual(record, [])


class TestBasetenDryRun(EnvGuardTest, TempDirTest):
    def setUp(self):
        EnvGuardTest.setUp(self)
        TempDirTest.setUp(self)

    def tearDown(self):
        EnvGuardTest.tearDown(self)

    def _report(self, **kwargs):
        image = make_image(self)
        kwargs.setdefault("ledger_path", self.path("usage.jsonl"))
        return prepare_baseten_probe(image, **kwargs)

    def test_offline_report_shape(self):
        os.environ[BASETEN_ENV_VAR] = FAKE_KEY
        os.environ["MODEL_API_KEY"] = OTHER_KEY
        report = self._report()
        self.assertFalse(report["transmitted"])
        self.assertEqual(report["provider"], "baseten")
        self.assertEqual(report["endpoint"],
                         "https://inference.baseten.co/v1/chat/completions")
        self.assertEqual(report["request"]["model"], BASETEN_MODEL_DEFAULT)
        self.assertEqual(report["request"]["max_tokens"],
                         BASETEN_MAX_TOKENS_DEFAULT)
        self.assertFalse(report["wire"]["has_image"])
        self.assertEqual(report["wire"]["message_content_types"], ["text"])
        dumped = json.dumps(report)
        self.assertNotIn("base64", dumped)
        self.assertNotIn("data:image", dumped)
        self.assertNotIn(FAKE_KEY, dumped)
        self.assertNotIn(OTHER_KEY, dumped)

    def test_writes_nothing(self):
        ledger = self.path("usage.jsonl")
        cache = self.path("cache")
        self._report()
        self.assertFalse(os.path.exists(ledger))
        self.assertFalse(os.path.exists(cache))

    def test_deterministic(self):
        first = self._report()
        second = self._report()
        self.assertEqual(first, second)

    def test_bad_image_and_cap_fail_closed(self):
        from reforge.probe import ProbeError
        with self.assertRaises(ProbeError):
            prepare_baseten_probe(self.path("missing.jpg"),
                                  ledger_path=self.path("u.jsonl"))
        image = make_image(self)
        with self.assertRaises(ProbeError):
            prepare_baseten_probe(image, max_tokens=0,
                                  ledger_path=self.path("u.jsonl"))

    def test_cli_dry_run_transmits_nothing(self):
        from reforge.cli import main
        image = make_image(self)
        ledger = self.path("usage.jsonl")
        rc = main(["baseten-dry-run", "--image", image,
                   "--ledger", ledger])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(ledger))


class TestMetaPathUnchanged(unittest.TestCase):
    def test_meta_transport_and_gate_intact(self):
        from reforge.http_transport import ResponsesHttpTransport
        from reforge.spark_request import send_spark_request
        transport = ResponsesHttpTransport(
            api_key=FAKE_KEY,
            request_fn=lambda request, timeout=None: (_ for _ in ()).throw(
                AssertionError("must not transmit")))
        self.assertEqual(transport.url,
                         "https://api.meta.ai/v1/responses")
        with self.assertRaises(LiveCallsDisabledError):
            send_spark_request(object(), transport=transport)


if __name__ == "__main__":
    unittest.main()
