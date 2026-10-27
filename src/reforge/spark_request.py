"""Offline preparation for a structured-geometry Spark request.

This module builds, dry-runs, and (only via explicitly injected transport)
sends a single request asking a vision model for a ``GeometryProposal``-
shaped JSON object. No transport is configured here, so nothing can reach
the network by default.

Unverified API facts (NOT established by repository code — do not rely on
them without a live check):

- Model id ``muse-spark-1.3``: operator-supplied; no code path has called it.
- Endpoint/auth: assumed to follow the SAM convention (OpenAI-compatible
  ``https://api.meta.ai/v1`` with ``MODEL_API_KEY``); unconfirmed for Spark.
- Image-input support, structured-output enforcement, per-call pricing, and
  usage/charge fields: all unknown. Post-call cost is therefore recorded as
  ``unknown`` (conservatively counted, never zero) until dashboard evidence
  says otherwise.

Design rules:

- The image and mask figures are *evidence in pixels*, never dimensions.
  The prompt states this explicitly and forbids the model from converting
  pixels to millimetres without a grounded scale.
- Unknown dimensions must come back null, with confidence 0.0 and a
  confirmation flag — never plausible guesses.
- Request construction, transport, parsing, validation, and CAD generation
  are separate stages. Parsing reuses :func:`parse_proposal_json` /
  :func:`validate_proposal`, so malformed or code-carrying output is
  rejected before anything downstream sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Callable, Mapping, Protocol

from .budget import BudgetGuard, BudgetGuardUnavailable, BudgetPolicy, ResultCache
from .geometry import (
    CAD_PRIMITIVE_MARKERS,
    SUPPORTED_CLASSIFICATIONS,
    parse_proposal_json,
    validate_proposal,
)
from .usage_ledger import UsageLedger, default_ledger_path, sha256_file

#: Operator-supplied default. NOT verified against any live endpoint.
SPARK_MODEL_DEFAULT = "muse-spark-1.3"
SPARK_OPERATION = "structured_geometry"

#: Request-template version. Bump whenever the instruction template, the
#: evidence framing, or any other cache-relevant request semantics change, so
#: responses cached under an older template can never satisfy a new lookup.
SPARK_CACHE_VERSION = 1

#: Oversized responses are rejected unparsed (schema caps lists/strings too).
MAX_RESPONSE_CHARS = 65536

#: Image MIME types the documented Responses image input accepts. The request
#: builder rejects anything else rather than letting the provider fail it.
SUPPORTED_IMAGE_MIMES = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp",
     "image/x-icon"}
)

#: Markers rejected in model output before JSON parsing. The CAD primitives
#: come from the schema's single screening policy so mappings (screened by
#: schema validation) and JSON strings (screened here first) face identical
#: screening; the two transport-specific extras stay local to this module.
CAD_CODE_MARKERS = CAD_PRIMITIVE_MARKERS + ("openscad", "#!")


class SparkRequestError(ValueError):
    """Raised when a request cannot be constructed (bad inputs, unsupported
    classification, missing image)."""


class LiveCallsDisabledError(RuntimeError):
    """Raised whenever transmission is attempted without an explicitly
    injected transport plus explicit opt-in. The default path never sends."""


class SparkTransportError(RuntimeError):
    """A transport-level failure. Recorded, never retried by this module."""


class SparkTransport(Protocol):
    """Boundary contract for transmitting a Responses API wire body.

    A transport receives the already-built wire body (a plain ``dict`` as
    produced by :func:`build_responses_body`) and returns the raw response
    envelope. It must not rebuild the request, must not retry, and must not
    fabricate usage data. Real network implementations live outside this
    module; no such implementation exists in this repository, and none is
    the default: ``send_spark_request`` without an explicitly injected
    transport raises :class:`LiveCallsDisabledError`.
    """

    def __call__(self, body: Mapping[str, Any]) -> Any:
        """Transmit one body; return the raw response envelope. Raises on
        connection, timeout, or HTTP errors; never returns a fabricated
        success."""
        ...  # pragma: no cover - interface only


@dataclass(frozen=True)
class SparkRequest:
    model: str
    image_path: str
    image_asset_hash: str
    image_size_px: tuple[int, int]
    concept: str
    classification: str
    mask_bounds_px: tuple[float, float, float, float] | None
    mask_fill_fraction: float | None
    interior_gap_count: int | None
    box_confidence: float | None
    instructions: str
    max_cost_usd: float | None


@dataclass(frozen=True)
class SendResult:
    status: str  # ok | error | blocked | cached
    detail: str
    proposal: Any = None
    from_cache: bool = False


def _evidence_figures(evidence: Any) -> dict:
    """Extract pixel figures from a MaskEvidence record or plain mapping."""
    if evidence is None:
        return {"bounds": None, "fill": None, "gaps": None, "confidence": None}
    if isinstance(evidence, Mapping):
        bounds = evidence.get("bounds_px")
        get = getattr
        return {
            "bounds": tuple(bounds) if bounds is not None else None,
            "fill": evidence.get("fill_fraction"),
            "gaps": evidence.get("interior_gap_count"),
            "confidence": evidence.get("box_confidence"),
        }
    try:
        bounds = evidence.bounds_px
    except AttributeError as exc:
        raise SparkRequestError(
            f"mask evidence must expose bounds_px/fill_fraction/"
            f"interior_gap_count/box_confidence, got {type(evidence).__name__}"
        ) from exc
    return {
        "bounds": tuple(bounds),
        "fill": evidence.fill_fraction,
        "gaps": evidence.interior_gap_count,
        "confidence": evidence.box_confidence,
    }


def build_instructions(classification: str, concept: str, figures: dict) -> str:
    """Build the model instruction text from the proposal contract."""
    bounds = figures["bounds"]
    bounds_text = (
        f"[{', '.join(f'{v:.1f}' for v in bounds)}]" if bounds is not None else "unknown")
    lines = [
        f"Identify the {concept} as a {classification} and return ONLY a JSON "
        f"object matching the geometry-proposal contract. No prose, no code.",
        "Coordinate frame (mandatory): origin at the plate corner (0, 0), "
        "x along length, y along width, z through thickness. Units: millimetres.",
        "Every measurement is an object with value, confidence (0-1), "
        "uncertainty_mm, source (observed_pixel | scale_estimate | "
        "model_inferred | user_confirmed), and requires_confirmation.",
        "Image-space evidence (PIXELS, not millimetres — never convert these "
        "to dimensions): "
        f"mask bounds {bounds_text}, fill "
        f"{figures['fill'] if figures['fill'] is not None else 'unknown'}, "
        f"interior gaps "
        f"{figures['gaps'] if figures['gaps'] is not None else 'unknown'}.",
        "Zero interior gaps in the mask does NOT prove the part has no holes; "
        "holes may be painted over by segmentation. Report only holes with "
        "visible evidence.",
        "Rules: unknown dimensions stay null with confidence 0.0 and "
        "requires_confirmation true — never fill them with plausible guesses. "
        "Confidence below 1.0 requires a positive uncertainty_mm. State every "
        "assumption, list everything unknown, flag everything needing user "
        "confirmation, and include a perspective-distortion warning.",
        "Forbidden: OpenSCAD, Python, shell, or any executable content. "
        "JSON data only.",
    ]
    return "\n".join(lines)


def build_spark_request(
    image_path: str,
    mask_evidence: Any = None,
    *,
    classification: str = "rectangular_plate",
    concept: str = "flat metal plate with circular holes",
    model: str = SPARK_MODEL_DEFAULT,
    max_cost_usd: float | None = None,
) -> SparkRequest:
    """Construct (but never send) a structured-geometry request."""
    if classification not in SUPPORTED_CLASSIFICATIONS:
        raise SparkRequestError(
            f"unsupported classification {classification!r}; this request layer "
            f"only prepares {sorted(SUPPORTED_CLASSIFICATIONS)}")
    if not concept or not concept.strip():
        raise SparkRequestError("concept must be a non-empty string")
    if not os.path.isfile(image_path):
        raise SparkRequestError(f"image not found: {image_path}")
    try:
        from PIL import Image

        with Image.open(image_path) as handle:
            image_size = (int(handle.width), int(handle.height))
    except Exception as exc:
        raise SparkRequestError(f"cannot read image {image_path}: {exc}") from exc
    figures = _evidence_figures(mask_evidence)
    return SparkRequest(
        model=model,
        image_path=image_path,
        image_asset_hash=sha256_file(image_path),
        image_size_px=image_size,
        concept=concept.strip(),
        classification=classification,
        mask_bounds_px=figures["bounds"],
        mask_fill_fraction=figures["fill"],
        interior_gap_count=figures["gaps"],
        box_confidence=figures["confidence"],
        instructions=build_instructions(classification, concept.strip(), figures),
        max_cost_usd=max_cost_usd,
    )


def build_responses_body(request: SparkRequest) -> dict:
    """Assemble the documented Responses API wire body as pure data.

    Shape (per the official image-understanding and structured-output
    documentation): ``model``, an ``input`` list with one user message
    holding ``input_text`` (the instructions) and ``input_image`` (a base64
    data URL), and ``text.format`` carrying the strict JSON Schema
    descriptor. Fully offline: reads the local image file into a transient
    base64 string that is returned to the caller and never persisted by this
    module — not in cache metadata, ledger entries, or logs.
    """
    import base64
    import mimetypes

    from .spark_schema import RESPONSE_FORMAT_DESCRIPTOR

    mime = mimetypes.guess_type(request.image_path)[0] or "image/jpeg"
    if mime not in SUPPORTED_IMAGE_MIMES:
        raise SparkRequestError(
            f"image MIME {mime!r} is not in the documented supported set "
            f"{sorted(SUPPORTED_IMAGE_MIMES)}")
    with open(request.image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return {
        "model": request.model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": request.instructions},
                {"type": "input_image",
                 "image_url": f"data:{mime};base64,{encoded}"},
            ],
        }],
        "text": {"format": dict(RESPONSE_FORMAT_DESCRIPTOR)},
    }


def parse_responses_usage(usage: Any) -> dict:
    """Parse the documented Responses ``usage`` object into token counts.

    Documented shape: ``{input_tokens, output_tokens, total_tokens,
    input_tokens_details: {cached_tokens}, output_tokens_details:
    {reasoning_tokens}}``. Missing, partial, or malformed input yields
    ``None`` per field (never invented, never zero-filled): accounting treats
    absent data as unknown. Reasoning tokens are a subset of the output total
    and are not reported separately.
    """
    result = {"input_tokens": None, "output_tokens": None,
              "total_tokens": None, "cached_tokens": None}
    if not isinstance(usage, Mapping):
        return result

    def number(value: Any) -> int | float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and value >= 0:
            return value
        return None

    for key in ("input_tokens", "output_tokens", "total_tokens"):
        result[key] = number(usage.get(key))
    details = usage.get("input_tokens_details")
    if isinstance(details, Mapping):
        result["cached_tokens"] = number(details.get("cached_tokens"))
    return result


def dry_run(
    request: SparkRequest,
    *,
    policy: BudgetPolicy | None = None,
    ledger_path: str | None = None,
) -> dict:
    """Describe what a live run WOULD do, without sending anything.

    Read-only against the real ledger (totals only, no entries written) and
    performs zero network calls. A dry run reporting "would allow" is not a
    transmission and never implies a request succeeded.
    """
    policy = policy or BudgetPolicy()
    ledger = UsageLedger(ledger_path or default_ledger_path())
    totals = ledger.totals()
    spent = (totals["confirmed_usd"] + totals["estimated_usd"]
             + totals["unknown_count"] * policy.unknown_cost_assumption_usd)
    discretionary = policy.total_limit_usd - policy.reserve_usd - spent
    bound = request.max_cost_usd
    if bound is None:
        bound = policy.per_experiment_cap_usd
    verdict, reason = "would-block", ""
    if bound > policy.per_experiment_cap_usd:
        reason = (f"bound ${bound:.2f} exceeds per-experiment cap "
                  f"${policy.per_experiment_cap_usd:.2f}")
    elif bound > discretionary:
        reason = (f"bound ${bound:.2f} exceeds discretionary ${discretionary:.2f} "
                  f"(reserve ${policy.reserve_usd:.2f} protected)")
    else:
        verdict, reason = "would-allow", "bound fits cap and discretionary budget"
    return {
        "action": "dry-run (no request sent)",
        "model": request.model,
        "operation": SPARK_OPERATION,
        "image_asset_hash": request.image_asset_hash,
        "image_size_px": list(request.image_size_px),
        "concept": request.concept,
        "classification": request.classification,
        "instruction_chars": len(request.instructions),
        "structured_output": "text.format json_schema strict (Responses API)",
        "bound_usd": bound,
        "bound_basis": ("operator-declared" if request.max_cost_usd is not None
                        else "policy per-experiment cap (default)"),
        "spent_usd": spent,
        "discretionary_usd": discretionary,
        "verdict": verdict,
        "reason": reason,
        "transport": "unconfigured (live calls disabled without injected transport)",
    }


def parse_spark_response(payload: Any):
    """Parse raw model output into a validated GeometryProposal.

    Rejects CAD/code-shaped replies before JSON parsing, then enforces the
    full schema. Raises :class:`ProposalError` for anything malformed,
    invalid, or code-carrying.
    """
    from .geometry import ProposalError, ProposalIssue

    if isinstance(payload, Mapping):
        return validate_proposal(payload)
    if not isinstance(payload, str):
        raise ProposalError([ProposalIssue(
            "structural", "<response>",
            f"model output must be JSON text or a mapping, "
            f"got {type(payload).__name__}")])
    lowered = payload.lower()
    for marker in CAD_CODE_MARKERS:
        if marker in lowered:
            raise ProposalError([ProposalIssue(
                "structural", "<response>",
                f"model output contains executable CAD/code marker {marker!r}; "
                f"only JSON data is accepted")])
    if len(payload) > MAX_RESPONSE_CHARS:
        raise ProposalError([ProposalIssue(
            "structural", "<response>",
            f"response is {len(payload)} chars, exceeding the "
            f"{MAX_RESPONSE_CHARS}-char limit (oversized)")])
    text = payload.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    return parse_proposal_json(text)


def _split_envelope(raw_response: Any) -> tuple[Any, Any]:
    """Split a transport response into (model output, usage object).

    Envelope convention for transports returning the documented response
    shape: {"output": <proposal text or mapping>, "usage": {...}}. Anything
    else is treated as the bare model output with no usage object.
    """
    if isinstance(raw_response, Mapping) and "output" in raw_response:
        return raw_response["output"], raw_response.get("usage")
    return raw_response, None


def _cache_key(request: SparkRequest) -> str:
    # The version leads the key so it is part of the hashed cache identity:
    # entries stored under another version live under another key entirely.
    return "|".join([f"request-template-v{SPARK_CACHE_VERSION}", request.model,
                     SPARK_OPERATION, request.image_asset_hash,
                     request.concept, request.classification])


def send_spark_request(
    request: SparkRequest,
    *,
    transport: SparkTransport | None = None,
    allow_live: bool = False,
    policy: BudgetPolicy | None = None,
    ledger_path: str | None = None,
    cache_dir: str | None = None,
    experiment_id: str = "spark-geometry-001",
    max_cost_usd: float | None = None,
) -> SendResult:
    """Transmit a prepared request through an explicitly injected transport.

    Disabled unless BOTH ``transport`` is provided AND ``allow_live`` is true;
    otherwise raises :class:`LiveCallsDisabledError` without touching the
    network, the guard, or the ledger. When enabled: single transmission, no
    retries; budget preflight first (a denial is ledgered, transport untouched);
    outcomes ledgered per project conventions with cost ``unknown`` unless the
    transport reports an explicit charge.
    """
    from .geometry import ProposalError

    if transport is None or not allow_live:
        raise LiveCallsDisabledError(
            "live Spark transmission is disabled in this increment: provide an "
            "explicit transport callable AND allow_live=True to send")
    policy = policy or BudgetPolicy()
    ledger = UsageLedger(ledger_path or default_ledger_path())
    cache = ResultCache(cache_dir or os.path.join(os.path.dirname(ledger.path), "cache"))
    bound = request.max_cost_usd if request.max_cost_usd is not None else max_cost_usd
    if bound is None:
        bound = policy.per_experiment_cap_usd

    key = _cache_key(request)
    cached = cache.get(key)
    if (isinstance(cached, Mapping) and cached.get("v") == 1
            and cached.get("request_version") == SPARK_CACHE_VERSION):
        try:
            cached_output, _ = _split_envelope(cached["response"])
            proposal = parse_spark_response(cached_output)
        except ProposalError as exc:
            ledger.log(
                provider="meta", model=request.model, operation=SPARK_OPERATION,
                status="error", asset_hash=request.image_asset_hash,
                experiment_id=experiment_id, image_count=1, cost_status="unknown",
                note=f"cached response unparseable ({exc}); no new request sent")
            return SendResult("error", f"cached response unparseable: {exc}")
        ledger.log(
            provider="meta", model=request.model, operation=SPARK_OPERATION,
            status="cached", asset_hash=request.image_asset_hash,
            experiment_id=experiment_id, image_count=1,
            cost_status="confirmed", charge_usd=0.0,
            note="reused cached response; no new request sent")
        return SendResult("cached", "reused cached response", proposal, from_cache=True)

    try:
        decision = BudgetGuard(policy, ledger).check(experiment_id, bound)
    except BudgetGuardUnavailable as exc:
        return SendResult("blocked", f"blocked: budget guard unavailable ({exc})")
    if not decision.allowed:
        return SendResult("blocked", f"blocked by budget guard: {decision.reason}")

    try:
        body = build_responses_body(request)
    except SparkRequestError as exc:
        ledger.log(
            provider="meta", model=request.model, operation=SPARK_OPERATION,
            status="error", asset_hash=request.image_asset_hash,
            experiment_id=experiment_id, image_count=1, cost_status="unknown",
            note=f"wire assembly failed ({exc}); nothing was transmitted, "
                 f"nothing billable")
        return SendResult("error", f"wire assembly failed: {exc}")
    try:
        raw_response = transport(body)  # exactly one transmission; no retries
    except LiveCallsDisabledError:
        raise
    except Exception as exc:
        ledger.log(
            provider="meta", model=request.model, operation=SPARK_OPERATION,
            status="error", asset_hash=request.image_asset_hash,
            experiment_id=experiment_id, image_count=1, cost_status="unknown",
            note=f"transmission failed: {type(exc).__name__}; cost unknown, "
                 f"may have been billed; no automatic retry")
        return SendResult("error", f"transmission failed: {type(exc).__name__}: {exc}")

    cache.put(key, {"v": 1, "request_version": SPARK_CACHE_VERSION,
                    "model": request.model,
                    "asset_hash": request.image_asset_hash,
                    "response": raw_response if isinstance(raw_response, (str, dict)) else ""})
    # Split before parsing so an enveloped response is validated by its
    # model-output content, never by the envelope itself.
    raw_response, envelope_usage = _split_envelope(raw_response)
    try:
        proposal = parse_spark_response(raw_response)
    except ProposalError as exc:
        ledger.log(
            provider="meta", model=request.model, operation=SPARK_OPERATION,
            status="error", asset_hash=request.image_asset_hash,
            experiment_id=experiment_id, image_count=1, cost_status="unknown",
            note=f"response unparseable/invalid ({exc}); cost unknown; no retry")
        return SendResult("error", f"response rejected: {exc}")

    usage = parse_responses_usage(envelope_usage)
    if isinstance(raw_response, Mapping):
        # Backward-compatible top-level reads for transports that inline
        # token or charge fields outside a usage object.
        for token_key in ("input_tokens", "output_tokens"):
            value = raw_response.get(token_key)
            if usage.get(token_key) is None and isinstance(value, (int, float)):
                usage[token_key] = value
        charge = raw_response.get("charge_usd", raw_response.get("cost_usd"))
        if isinstance(charge, (int, float)):
            usage["charge_usd"] = float(charge)
    note = "structured geometry received and schema-validated"
    if usage.get("cached_tokens") is not None:
        note += f"; cached_input_tokens={usage['cached_tokens']}"
    ledger.log(
        provider="meta", model=request.model, operation=SPARK_OPERATION,
        status="ok", asset_hash=request.image_asset_hash,
        experiment_id=experiment_id,
        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
        image_count=1, charge_usd=usage.get("charge_usd"),
        cost_status="confirmed" if "charge_usd" in usage else "unknown",
        cost_basis=("transport-reported charge" if "charge_usd" in usage else
                    "transport reported no charge or usage; pricing unverified — "
                    "reconcile against provider dashboard"),
        note=note)
    return SendResult("ok", "structured geometry received and validated", proposal)
