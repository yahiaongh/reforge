"""HTTP transport tests. Fully offline: injected fake openers and socket
blocking only. No live calls, no real credentials, no network."""

import io
import json
import os
import socket
import unittest
import urllib.error
from unittest import mock

from tests import TempDirTest
from reforge.http_transport import ResponsesHttpTransport
from reforge.sam_experiment import MissingCredentialsError
from reforge.spark_request import (
    LiveCallsDisabledError,
    SparkTransportError,
    build_spark_request,
    dry_run,
    send_spark_request,
)

try:
    from test_geometry import valid_proposal_dict
except ImportError:  # run as tests.test_http_transport instead of discover
    from tests.test_geometry import valid_proposal_dict

FAKE_KEY = "sk-test-SECRET-VALUE-xyz"


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def opener_for(payload=None, error=None):
    calls = []

    def opener(request, timeout=None):
        calls.append({"url": request.full_url,
                      "headers": dict(request.header_items()),
                      "data": request.data,
                      "timeout": timeout})
        if error is not None:
            raise error
        return FakeHTTPResponse(payload)

    opener.calls = calls
    return opener


def envelope_envelope(text, usage=None):
    envelope = {"output": [{"type": "message", "content": [
        {"type": "output_text", "text": text}]}]}
    if usage is not None:
        envelope["usage"] = usage
    return json.dumps(envelope).encode("utf-8")


def make_request(case: TempDirTest):
    from PIL import Image

    image = case.path("part.jpg")
    Image.new("RGB", (64, 48), (180, 180, 180)).save(image)
    return build_spark_request(image, None)


def send_kwargs(case: TempDirTest):
    return {"ledger_path": case.path("usage.jsonl"),
            "cache_dir": case.path("cache"),
            "experiment_id": "test-http-transport"}


class TestTransportConstruction(TempDirTest):
    def test_missing_key_rejected(self):
        with self.assertRaises(MissingCredentialsError):
            ResponsesHttpTransport(api_key="")

    def test_non_https_base_rejected(self):
        with self.assertRaises(ValueError):
            ResponsesHttpTransport(api_key="x", base_url="http://x.test/v1")

    def test_nonpositive_timeout_rejected(self):
        with self.assertRaises(ValueError):
            ResponsesHttpTransport(api_key="x", timeout_s=0)

    def test_repr_hides_key(self):
        transport = ResponsesHttpTransport(api_key=FAKE_KEY)
        self.assertNotIn("SECRET", repr(transport))
        self.assertIn("https://api.meta.ai/v1/responses", repr(transport))

    def test_from_env_missing_key(self):
        old = os.environ.pop("MODEL_API_KEY", None)
        self.addCleanup(
            lambda: os.environ.update({"MODEL_API_KEY": old}) if old else None)
        with self.assertRaises(MissingCredentialsError):
            ResponsesHttpTransport.from_env(request_fn=lambda r, timeout=None: None)

    def test_from_env_reads_key_only_when_called(self):
        os.environ["MODEL_API_KEY"] = FAKE_KEY
        self.addCleanup(lambda: os.environ.pop("MODEL_API_KEY", None))
        transport = ResponsesHttpTransport.from_env(
            request_fn=lambda r, timeout=None: None)
        self.assertNotIn("SECRET", repr(transport))


class TestWireBehavior(TempDirTest):
    def test_posts_json_to_responses_endpoint(self):
        opener = opener_for(envelope_envelope(json.dumps(valid_proposal_dict())))
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        body = {"model": "muse-spark-1.3", "input": [], "text": {}}
        transport(body)
        self.assertEqual(len(opener.calls), 1)
        call = opener.calls[0]
        self.assertEqual(call["url"], "https://api.meta.ai/v1/responses")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {FAKE_KEY}")
        self.assertEqual(call["headers"]["Content-type"], "application/json")
        self.assertEqual(json.loads(call["data"].decode("utf-8")), body)
        self.assertEqual(call["timeout"], 120.0)

    def test_exactly_one_request_no_retry(self):
        opener = opener_for(error=urllib.error.URLError("down"))
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        with self.assertRaises(SparkTransportError):
            transport({"model": "m"})
        self.assertEqual(len(opener.calls), 1)

    def test_timeout_maps_to_transport_error(self):
        opener = opener_for(error=socket.timeout("timed out"))
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        with self.assertRaises(SparkTransportError) as ctx:
            transport({"model": "m"})
        self.assertIn("timeout", str(ctx.exception).lower())
        self.assertNotIn("SECRET", str(ctx.exception))

    def test_connection_error_maps(self):
        opener = opener_for(error=ConnectionRefusedError("refused"))
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        with self.assertRaises(SparkTransportError):
            transport({"model": "m"})

    def test_http_errors_carry_status_not_body(self):
        for code in (400, 429, 500):
            fp = io.BytesIO(b'{"error": {"message": "bad", "key": "SECRET-XYZ"}}')
            opener = opener_for(error=urllib.error.HTTPError(
                "https://api.meta.ai/v1/responses", code, "Err", {}, fp))
            transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
            with self.assertRaises(SparkTransportError) as ctx:
                transport({"model": "m"})
            self.assertIn(str(code), str(ctx.exception))
            self.assertNotIn("SECRET", str(ctx.exception))

    def test_malformed_json_envelope_errors(self):
        opener = opener_for(b"not json {{{")
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        with self.assertRaises(SparkTransportError):
            transport({"model": "m"})

    def test_unexpected_envelope_shape_errors(self):
        opener = opener_for(b"[1, 2, 3]")
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        with self.assertRaises(SparkTransportError):
            transport({"model": "m"})

    def test_missing_output_text_extracts_none(self):
        opener = opener_for(json.dumps({"usage": {}}).encode())
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        self.assertEqual(transport({"model": "m"}),
                         {"output": None, "usage": {}})

    def test_usage_passes_through_untouched(self):
        usage = {"input_tokens": 100, "output_tokens": 20,
                 "input_tokens_details": {"cached_tokens": 5}}
        opener = opener_for(json.dumps({"output": [], "usage": usage}).encode())
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        self.assertEqual(transport({"model": "m"})["usage"], usage)


class TestSendIntegration(TempDirTest):
    def test_end_to_end_ok_with_usage(self):
        from reforge.usage_ledger import UsageLedger

        payload = envelope_envelope(
            json.dumps(valid_proposal_dict()),
            {"input_tokens": 1200, "output_tokens": 300,
             "input_tokens_details": {"cached_tokens": 200}})
        opener = opener_for(payload)
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        result = send_spark_request(
            make_request(self), transport=transport, allow_live=True,
            **send_kwargs(self))
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.proposal.holes), 4)
        entry = UsageLedger(self.path("usage.jsonl")).entries()[0]
        self.assertEqual((entry.input_tokens, entry.output_tokens), (1200, 300))
        self.assertIn("cached_input_tokens=200", entry.note)
        self.assertNotIn("SECRET", entry.note)

    def test_invalid_proposal_text_fails_closed(self):
        payload = envelope_envelope("difference() { cube([1]); }")
        opener = opener_for(payload)
        transport = ResponsesHttpTransport(api_key=FAKE_KEY, request_fn=opener)
        result = send_spark_request(
            make_request(self), transport=transport, allow_live=True,
            **send_kwargs(self))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.proposal)

    def test_socket_blocked_real_opener_fails_closed(self):
        old = os.environ.pop("MODEL_API_KEY", None)
        self.addCleanup(
            lambda: os.environ.update({"MODEL_API_KEY": old}) if old else None)
        transport = ResponsesHttpTransport(api_key=FAKE_KEY)  # real opener
        with mock.patch("socket.create_connection",
                         side_effect=AssertionError("no network")):
            with mock.patch("socket.socket",
                             side_effect=AssertionError("no network")):
                result = send_spark_request(
                    make_request(self), transport=transport, allow_live=True,
                    **send_kwargs(self))
        self.assertEqual(result.status, "error")

    def test_disabled_path_touches_nothing(self):
        request = make_request(self)
        with self.assertRaises(LiveCallsDisabledError):
            send_spark_request(request, **send_kwargs(self))
        self.assertFalse(os.path.exists(self.path("usage.jsonl")))

    def test_dry_run_needs_no_key(self):
        old = os.environ.pop("MODEL_API_KEY", None)
        self.addCleanup(
            lambda: os.environ.update({"MODEL_API_KEY": old}) if old else None)
        report = dry_run(make_request(self),
                         ledger_path=self.path("usage.jsonl"))
        self.assertEqual(report["verdict"], "would-allow")
        self.assertFalse(os.path.exists(self.path("usage.jsonl")))


if __name__ == "__main__":
    unittest.main()
