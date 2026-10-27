"""Baseten Chat Completions transport (disabled by default, offline-safe).

Separate from the Meta-hosted Responses API path
(:mod:`reforge.spark_request` / :mod:`reforge.http_transport`): different
base URL, different endpoint shape, different credential, different cache
version. Nothing here is shared with the Meta transport except the
house conventions it mirrors — injected request function, single
transmission, no retries, key-private errors, guard-first execution.

Target (operator-supplied, not independently verified):
model ``meta-models/muse-spark-open-1026`` via
``POST https://inference.baseten.co/v1/chat/completions`` with
``Authorization: Bearer <BASETEN_API_KEY>``.

Unresolved provider facts (do NOT assume otherwise):
- whether the slug exists on the operator's account and what it costs;
- whether a base64 ``data:`` image URL is accepted (Baseten documents
  server-fetched HTTPS URLs; data-URL delivery is UNVERIFIED);
- the exact usage-field shape (OpenAI-compatible ``prompt_tokens`` /
  ``completion_tokens`` / ``total_tokens`` assumed, verified nowhere).

Consequences enforced by this module:
- image content rides only behind an explicit delivery approval, and a
  data URL is always labelled unverified;
- ``max_tokens`` bounds output length only. It is NOT a dollar cap: input
  tokens (prompt + image) also bill, and no per-token price is known, so
  the budget bound stays an operator-declared policy tripwire, never a
  provider guarantee;
- mocked/fake transports prove offline behavior only, never provider
  compatibility.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .sam_experiment import MissingCredentialsError
from .spark_request import (
    LiveCallsDisabledError,
    SparkRequestError,
    SparkTransportError,
)


BASETEN_BASE_URL_DEFAULT = "https://inference.baseten.co/v1"
BASETEN_CHAT_PATH = "/chat/completions"
BASETEN_MODEL_DEFAULT = "meta-models/muse-spark-open-1026"
BASETEN_ENV_VAR = "BASETEN_API_KEY"
BASETEN_OPERATION = "chat_structured_geometry"
BASETEN_TIMEOUT_S_DEFAULT = 120.0
BASETEN_MAX_TOKENS_DEFAULT = 1500
# Independent cache namespace from the Meta Responses path: entries stored
# under one version are never valid for another.
BASETEN_CACHE_VERSION = 1
DATA_URL_UNVERIFIED_NOTE = (
    "data: image delivery is UNVERIFIED against Baseten docs, which "
    "describe server-fetched HTTPS URLs; a data URL may be rejected "
    "or billed differently")
COST_NOTE = (
    "max_tokens bounds output length only, not total cost: input/prompt "
    "tokens (including any image) also bill, and no per-token price is "
    "known for this model. The budget bound is an operator-declared "
    "policy tripwire, never a provider-enforced dollar cap.")


@dataclass(frozen=True)
class BasetenAuthorization:
    """Explicit live-transmission prerequisites for one Baseten probe.

    Every field defaults to deny. ``max_bound_usd`` is the operator's
    declared policy bound for this probe (see COST_NOTE); ``None`` with no
    ``manual_exception`` fails the budget preflight. Nothing here reads
    credentials or contacts any provider.
    """
    allow_live: bool = False
    max_bound_usd: float | None = None
    manual_exception: str = ""
    image_delivery_approved: bool = False


@dataclass(frozen=True)
class BasetenResult:
    """Outcome of a Baseten send attempt. ``proposal`` is set only after
    the existing local validators accept the model output."""
    status: str  # ok | error | blocked
    detail: str
    proposal: Any = None
    output_text: str | None = None
    usage: dict | None = None
    finish_reason: str | None = None


def build_baseten_body(
    *,
    model: str = BASETEN_MODEL_DEFAULT,
    instructions: str,
    max_tokens: int = BASETEN_MAX_TOKENS_DEFAULT,
    image_data_url: str | None = None,
    response_format: Mapping[str, Any] | None = None,
) -> dict:
    """Assemble a Chat Completions wire body. Offline: no I/O, no key.

    ``response_format`` defaults to JSON mode (``{"type": "json_object"}``,
    the documented option that needs no schema-subset gamble). An explicit
    ``json_schema`` override is passed through untouched but stays
    untrusted: local validation remains authoritative either way.
    """
    if not isinstance(model, str) or not model:
        raise SparkRequestError("Baseten model slug must be a non-empty string")
    if not isinstance(instructions, str) or not instructions.strip():
        raise SparkRequestError("Baseten instructions must be non-empty text")
    if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
            or max_tokens <= 0):
        raise SparkRequestError(
            f"max_tokens must be a positive integer, got {max_tokens!r}")
    if response_format is None:
        response_format = {"type": "json_object"}
    if not isinstance(response_format, Mapping) or "type" not in response_format:
        raise SparkRequestError(
            "response_format must be a mapping with a 'type' field")
    content: list[dict] = [{"type": "text", "text": instructions}]
    if image_data_url is not None:
        if (not isinstance(image_data_url, str)
                or not (image_data_url.startswith("data:")
                        or image_data_url.startswith("https://"))):
            raise SparkRequestError(
                "image must be a data: URL (UNVERIFIED compatibility) or an "
                "https:// URL the provider can fetch; nothing else is accepted")
        content.append({"type": "image_url",
                        "image_url": {"url": image_data_url}})
    return {"model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "response_format": dict(response_format)}


def body_has_image(body: Mapping[str, Any]) -> bool:
    """True when the wire body carries any image_url part."""
    try:
        content = body["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return False
    return any(isinstance(part, Mapping) and part.get("type") == "image_url"
               for part in content)


def parse_baseten_usage(usage: Any) -> dict | None:
    """Read an OpenAI-compatible usage object conservatively.

    Returns token counts with ``None`` for missing fields (missing usage is
    NOT zero usage) and ``None`` for malformed objects. Never produces a
    monetary cost: tokens are not charges.
    """
    if usage is None:
        return None
    if not isinstance(usage, Mapping):
        return None

    def _count(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return value
        return None

    counts = {"input_tokens": _count(usage.get("prompt_tokens")),
              "output_tokens": _count(usage.get("completion_tokens")),
              "total_tokens": _count(usage.get("total_tokens"))}
    if all(value is None for value in counts.values()):
        return None
    return counts


class BasetenChatTransport:
    """Single-shot Chat Completions POST. Testable via an injected request
    function; the default is the real ``urllib`` opener — but nothing in
    this repository constructs it with the default outside an explicitly
    authorized live path, and no CLI command does so at all."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = BASETEN_BASE_URL_DEFAULT,
        timeout_s: float = BASETEN_TIMEOUT_S_DEFAULT,
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
        self._url = base_url.rstrip("/") + BASETEN_CHAT_PATH
        self._timeout_s = float(timeout_s)
        self._request_fn = request_fn or urllib.request.urlopen

    def __repr__(self) -> str:  # never leaks the key
        return (f"BasetenChatTransport(url={self._url!r}, "
                f"timeout_s={self._timeout_s:g})")

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str = BASETEN_BASE_URL_DEFAULT,
        timeout_s: float = BASETEN_TIMEOUT_S_DEFAULT,
        request_fn: Callable | None = None,
    ) -> "BasetenChatTransport":
        """Build a transport from the environment. Reads ``BASETEN_API_KEY``
        ONLY — never ``MODEL_API_KEY`` or any other provider's credential,
        so the two transports cannot cross-activate."""
        api_key = os.environ.get(BASETEN_ENV_VAR)
        if not api_key:
            raise MissingCredentialsError(f"{BASETEN_ENV_VAR} is not set")
        return cls(api_key=api_key, base_url=base_url,
                   timeout_s=timeout_s, request_fn=request_fn)

    @property
    def url(self) -> str:
        return self._url

    def __call__(self, body: Mapping[str, Any]) -> dict:
        try:
            payload = json.dumps(dict(body)).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SparkTransportError(
                f"request body is not JSON-serializable: {exc}")
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
        return _normalize_chat_envelope(envelope)


def _message_text(message: Any) -> str | None:
    """Extract assistant text from a Chat Completions message."""
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        texts = [block.get("text") for block in content
                 if isinstance(block, Mapping)
                 and block.get("type") == "text"
                 and isinstance(block.get("text"), str)
                 and block.get("text")]
        return "\n".join(texts) if texts else None
    return None


def _normalize_chat_envelope(envelope: Mapping[str, Any]) -> dict:
    """Normalize to ``{"output", "usage", "finish_reason"}``.

    Missing choices, missing message content, and malformed shapes raise:
    an absent proposal is an error, never an empty success. Truncation
    (``finish_reason == "length"``) is reported in the envelope and fails
    closed in :func:`send_baseten_request`, since cut-off JSON is unusable.
    Usage passes through untouched for the conservative accounting path.
    """
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SparkTransportError("response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise SparkTransportError("response choice is not an object")
    text = _message_text(first.get("message"))
    if text is None:
        raise SparkTransportError("response choice has no message content")
    return {"output": text,
            "usage": envelope.get("usage"),
            "finish_reason": first.get("finish_reason")}


def send_baseten_request(
    body: Mapping[str, Any],
    *,
    transport: Any = None,
    auth: BasetenAuthorization | None = None,
    image_asset_hash: str = "",
    policy: Any = None,
    ledger_path: str | None = None,
    experiment_id: str = "baseten-geometry-001",
    max_cost_usd: float | None = None,
) -> BasetenResult:
    """Guarded single transmission through an explicitly injected transport.

    Denial order (every denial happens BEFORE the request function runs):
    1. no transport or ``allow_live`` false → :class:`LiveCallsDisabledError`
       (no ledger touch, mirroring the Meta path);
    2. invalid model/timeout/cap configuration → ``blocked``;
    3. budget preflight via the existing :class:`BudgetGuard` → ``blocked``.
       The bound is the operator-declared ``max_cost_usd``/``max_bound_usd``
       policy tripwire (see COST_NOTE); ``None`` with no manual exception
       blocks, exactly like the Meta path;
    4. body carries image content without ``image_delivery_approved`` →
       ``blocked`` (unverified data-URL compatibility must be accepted
       explicitly, never assumed).

    On success the model text still passes the full existing pipeline —
    CAD-marker pre-scan plus :func:`parse_spark_response` validation — and a
    proposal is returned only if all of it passes. Exactly one transmission,
    no retries. Outcomes are ledgered with cost ``unknown`` (tokens are not
    charges; reconcile against the Baseten dashboard).
    """
    from .budget import BudgetGuard, BudgetGuardUnavailable, BudgetPolicy
    from .geometry import ProposalError
    from .spark_request import parse_spark_response
    from .usage_ledger import UsageLedger, default_ledger_path

    auth = auth or BasetenAuthorization()
    if transport is None or not auth.allow_live:
        raise LiveCallsDisabledError(
            "live Baseten transmission is disabled: provide an explicit "
            "transport AND BasetenAuthorization(allow_live=True) to send")
    policy = policy or BudgetPolicy()
    ledger = UsageLedger(ledger_path or default_ledger_path())
    model = body.get("model") if isinstance(body, Mapping) else None
    if not isinstance(model, str) or not model:
        _log_blocked(ledger, model="", experiment_id=experiment_id,
                      note="blocked: invalid or missing model slug")
        return BasetenResult("blocked", "blocked: invalid or missing model slug")
    max_tokens = body.get("max_tokens") if isinstance(body, Mapping) else None
    if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
            or max_tokens <= 0):
        _log_blocked(ledger, model=model, experiment_id=experiment_id,
                      note="blocked: max_tokens output cap is missing or invalid")
        return BasetenResult(
            "blocked", "blocked: max_tokens output cap is missing or invalid")
    bound = max_cost_usd if max_cost_usd is not None else auth.max_bound_usd
    try:
        decision = BudgetGuard(policy, ledger).check(
            experiment_id, bound, manual_exception=auth.manual_exception)
    except BudgetGuardUnavailable as exc:
        return BasetenResult(
            "blocked", f"blocked: budget guard unavailable ({exc})")
    if not decision.allowed:
        return BasetenResult(
            "blocked", f"blocked by budget guard: {decision.reason}")
    if body_has_image(body) and not auth.image_delivery_approved:
        _log_blocked(
            ledger, model=model, experiment_id=experiment_id,
            note="blocked: body carries image content but image delivery "
                 "was not explicitly approved (data-URL compatibility "
                 "unverified)")
        return BasetenResult(
            "blocked",
            "blocked: image delivery not explicitly approved for this probe")
    try:
        envelope = transport(body)  # exactly one transmission; no retries
    except LiveCallsDisabledError:
        raise
    except Exception as exc:
        ledger.log(
            provider="baseten", model=model, operation=BASETEN_OPERATION,
            status="error", asset_hash=image_asset_hash,
            experiment_id=experiment_id, image_count=1 if body_has_image(body) else 0,
            cost_status="unknown",
            note=f"transmission failed: {type(exc).__name__}; cost unknown, "
                 f"may have been billed; no automatic retry")
        return BasetenResult(
            "error", f"transmission failed: {type(exc).__name__}: {exc}")
    if not isinstance(envelope, Mapping) or envelope.get("output") is None:
        ledger.log(
            provider="baseten", model=model, operation=BASETEN_OPERATION,
            status="error", asset_hash=image_asset_hash,
            experiment_id=experiment_id,
            image_count=1 if body_has_image(body) else 0,
            cost_status="unknown",
            note="transport returned no usable output; cost unknown; no retry")
        return BasetenResult("error", "transport returned no usable output")
    if envelope.get("finish_reason") == "length":
        ledger.log(
            provider="baseten", model=model, operation=BASETEN_OPERATION,
            status="error", asset_hash=image_asset_hash,
            experiment_id=experiment_id,
            image_count=1 if body_has_image(body) else 0,
            cost_status="unknown",
            note="response truncated by max_tokens (finish_reason=length); "
                 "cut-off JSON is unusable; cost unknown; no retry")
        return BasetenResult(
            "error", "response truncated by max_tokens; unusable",
            output_text=envelope.get("output"),
            finish_reason="length")
    try:
        proposal = parse_spark_response(envelope.get("output"))
    except ProposalError as exc:
        ledger.log(
            provider="baseten", model=model, operation=BASETEN_OPERATION,
            status="error", asset_hash=image_asset_hash,
            experiment_id=experiment_id,
            image_count=1 if body_has_image(body) else 0,
            cost_status="unknown",
            note=f"response unparseable/invalid ({exc}); cost unknown; no retry")
        return BasetenResult(
            "error", f"response rejected: {exc}",
            output_text=envelope.get("output"),
            finish_reason=envelope.get("finish_reason"))
    usage = parse_baseten_usage(
        envelope.get("usage") if isinstance(envelope, Mapping) else None)
    ledger.log(
        provider="baseten", model=model, operation=BASETEN_OPERATION,
        status="ok", asset_hash=image_asset_hash,
        experiment_id=experiment_id,
        input_tokens=usage.get("input_tokens") if usage else None,
        output_tokens=usage.get("output_tokens") if usage else None,
        image_count=1 if body_has_image(body) else 0,
        cost_status="unknown",
        cost_basis="Baseten reported no monetary charge; token counts are "
                    "not charges — reconcile against the Baseten dashboard",
        note="chat structured geometry received and locally validated")
    return BasetenResult(
        "ok", "chat structured geometry received and validated", proposal,
        output_text=envelope.get("output"), usage=usage,
        finish_reason=envelope.get("finish_reason"))


def _log_blocked(ledger: Any, *, model: str, experiment_id: str,
                 note: str) -> None:
    """Record a pre-transmission denial. Never raises: a logging failure
    must not convert a denial into an accidental send."""
    try:
        ledger.log(
            provider="baseten", model=model, operation=BASETEN_OPERATION,
            status="blocked", experiment_id=experiment_id,
            cost_status="unknown", note=note)
    except OSError:
        pass
