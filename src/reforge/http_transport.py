"""Production-quality Responses API HTTP transport (disabled by default).

Implements the :class:`SparkTransport` boundary with the standard library
only (``urllib`` — no new dependencies): exactly one POST per invocation,
finite configurable timeout, no retries, and fail-closed error mapping.
Nothing in this repository instantiates it outside tests, no CLI flag or
environment variable enables it implicitly, and :func:`send_spark_request`
still refuses to transmit without an explicitly injected transport plus
explicit opt-in.

Credential discipline: the key travels only as an explicit constructor
argument (or via :meth:`from_env`, which is the single place that reads
``MODEL_API_KEY``), is stored in a private attribute, and never appears in
``repr``, exceptions, logs, ledger entries, or cache records.

Response handling: the raw envelope is returned normalized as
``{"output": <extracted text or None>, "usage": <usage object or None>}``.
Output text is extracted from message items carrying ``output_text``/text
content (the documented accessor shape); anything else yields ``None`` and
fails closed downstream. Usage passes through untouched for the existing
accounting path, which still records cost as unknown unless a charge is
reported.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from .sam_experiment import MissingCredentialsError, SAM_BASE_URL
from .spark_request import SparkTransport, SparkTransportError

RESPONSES_PATH = "/responses"
DEFAULT_TIMEOUT_S = 120.0


class ResponsesHttpTransport:
    """Single-shot Responses API POST transport. Testable via an injected
    request function; the default is the real ``urllib`` opener."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = SAM_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        request_fn: Callable | None = None,
    ):
        if not isinstance(api_key, str) or not api_key:
            raise MissingCredentialsError(
                "a non-empty API key string is required to build a transport")
        if not isinstance(base_url, str) or not base_url.startswith("https://"):
            raise ValueError(f"base_url must be an https:// URL, got {base_url!r}")
        if not isinstance(timeout_s, (int, float)) or not timeout_s > 0:
            raise ValueError(f"timeout_s must be a positive finite number, "
                             f"got {timeout_s!r}")
        self._api_key = api_key
        self._url = base_url.rstrip("/") + RESPONSES_PATH
        self._timeout_s = float(timeout_s)
        self._request_fn = request_fn or urllib.request.urlopen

    def __repr__(self) -> str:  # never leaks the key
        return (f"ResponsesHttpTransport(url={self._url!r}, "
                f"timeout_s={self._timeout_s:g})")

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str = SAM_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        request_fn: Callable | None = None,
    ) -> "ResponsesHttpTransport":
        """Build a transport from the environment. This is the ONLY place
        that reads ``MODEL_API_KEY``; nothing else in the codebase does."""
        api_key = os.environ.get("MODEL_API_KEY")
        if not api_key:
            raise MissingCredentialsError(
                "MODEL_API_KEY is not set in the environment")
        return cls(api_key=api_key, base_url=base_url,
                   timeout_s=timeout_s, request_fn=request_fn)

    @property
    def url(self) -> str:
        return self._url

    def __call__(self, body: Mapping[str, Any]) -> dict:
        try:
            payload = json.dumps(dict(body)).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SparkTransportError(f"request body is not JSON-serializable: {exc}")
        # The key lives only on this transient Request object, which is
        # never logged, persisted, or interpolated into error text below.
        request = urllib.request.Request(
            self._url, data=payload, method="POST",
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"})
        try:
            with self._request_fn(request, timeout=self._timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise SparkTransportError(
                f"provider returned HTTP {exc.code} ({exc.reason})") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError, OSError) as exc:
            raise SparkTransportError(
                f"connection/timeout failure: {type(exc).__name__}") from exc
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise SparkTransportError(
                f"provider returned malformed JSON: {exc}") from exc
        if not isinstance(envelope, Mapping):
            raise SparkTransportError(
                f"unexpected response shape: {type(envelope).__name__}")
        return {"output": _extract_output_text(envelope),
                "usage": envelope.get("usage")}


def _extract_output_text(envelope: Mapping[str, Any]) -> str | None:
    """Pull assistant text from a Responses output array.

    Walks ``output[]`` message items for ``output_text``/``text`` content
    (the documented accessor shape). Returns ``None`` when no text is found
    so downstream parsing fails closed instead of guessing.
    """
    output = envelope.get("output")
    if not isinstance(output, list):
        return None
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            # Some SDK projections join text at the item level.
            direct = item.get("text")
            if isinstance(direct, str) and direct:
                texts.append(direct)
            continue
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") in ("output_text", "text"):
                text = block.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
    return "\n".join(texts) if texts else None
