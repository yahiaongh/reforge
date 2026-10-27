"""Single-purpose, budget-guarded SAM 3.1 segmentation experiment.

This module is the ONLY sanctioned path to a paid SAM request. The legacy
``verify_sam3.py`` script delegates here, so no entry point can reach the
network without passing the :class:`BudgetGuard` first.

Enforced properties:

- The guard is checked immediately before the paid operation, in the same
  process. A ``blocked`` decision means no client is built and no request
  is sent.
- At most ONE ``responses.create`` call per run. There is no retry loop;
  a retry is a new run with ``is_retry=True`` accounting.
- Every outcome is ledgered: ``ok`` (with tokens/charges when the provider
  returns them, else cost ``unknown``), ``error`` (API/stream failure after
  send, cost ``unknown`` — it may have been billed), ``blocked`` (never
  sent, bills nothing), ``cached`` (valid prior result reused, no new call).
- The ledger never receives secrets, prompt text, or image bytes — only the
  model id, operation, image SHA-256, counts, and cost fields.
- Cost honesty: ``max_cost_usd`` is an operator-declared *policy* bound
  (default: the per-experiment cap). It is NOT a provider-side cap. The
  provider's per-call price for SAM segmentation is not published to this
  module, so post-call cost is ``unknown`` unless the response carries an
  explicit charge. Reconcile against the provider dashboard.

Segmentation yields image-space geometry only — never millimetres.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from typing import Any, Callable

from .budget import BudgetGuard, BudgetGuardUnavailable, BudgetPolicy, ResultCache
from .usage_ledger import UsageLedger, default_ledger_path, sha256_file

SAM_MODEL = "sam-3.1"
SAM_OPERATION = "image_segmentation"
SAM_BASE_URL = "https://api.meta.ai/v1"
CACHE_VERSION = 1


class MissingCredentialsError(RuntimeError):
    """Raised before any network use when no API key is available."""


@dataclass
class ExperimentResult:
    status: str  # ok | error | blocked | cached
    detail: str
    overlay_path: str = ""
    from_cache: bool = False
    box_count: int = 0
    mask_count: int = 0
    summary: dict = field(default_factory=dict)


def _default_client_factory():
    """Build the live API client. Reads only the key's presence here; the key
    value never leaves the client object and is never logged."""
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise MissingCredentialsError(
            "MODEL_API_KEY is not set in the environment. "
            "Export it (env only, never commit it) before running the experiment."
        )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise MissingCredentialsError(
            "the 'openai' package is required for live calls "
            "(pip install -e .[models])"
        ) from exc
    return AsyncOpenAI(base_url=SAM_BASE_URL, api_key=api_key)


def _default_decode_mask(mask) -> bytes:
    from meta_sam_parser import decode_mask_to_raster

    return decode_mask_to_raster(mask)


def _default_parse_stream(events):
    from meta_sam_parser import image_segmentation_format, parse_responses_stream

    return parse_responses_stream(events, image_segmentation_format())


def _record_to_dict(record) -> dict:
    if record.kind == "box":
        return {
            "kind": "box",
            "object_id": record.object_id,
            "left": float(record.left),
            "top": float(record.top),
            "right": float(record.right),
            "bottom": float(record.bottom),
            "confidence": record.confidence,
        }
    if record.kind == "mask":
        return {
            "kind": "mask",
            "object_id": record.object_id,
            "bounds": {
                "left": float(record.bounds.left),
                "top": float(record.bounds.top),
                "right": float(record.bounds.right),
                "bottom": float(record.bounds.bottom),
            },
            "mask": {
                "encoding": record.mask.encoding,
                "payload": record.mask.payload,
                "width": int(record.mask.width),
                "height": int(record.mask.height),
            },
            "confidence": record.confidence,
        }
    return {"kind": getattr(record, "kind", "unknown")}


def _extract_usage(result) -> dict:
    """Best-effort usage/cost extraction. Returns {} when the provider exposes
    nothing machine-readable (the normal SAM case → cost stays unknown)."""
    usage: dict[str, Any] = {}
    for attr in ("usage",):
        candidate = getattr(result, attr, None)
        if candidate is None:
            continue
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(candidate, key, None)
            if isinstance(value, (int, float)):
                usage[key] = value
    outcome = getattr(result, "outcome", None)
    for key in ("charge_usd", "cost_usd", "price_usd"):
        value = getattr(outcome, key, None)
        if isinstance(value, (int, float)):
            usage["charge_usd"] = float(value)
    return usage


def render_overlay(image_path: str, summary: dict, out_path: str,
                   decode_mask: Callable | None = None) -> str:
    """Regenerate the red-mask overlay from a cached or live summary. Local
    only; needs no credentials and makes no calls."""
    from PIL import Image
    import numpy as np

    decode = decode_mask or _default_decode_mask
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    class _Bounds:
        def __init__(self, d):
            self.__dict__.update(d)

    class _Mask:
        def __init__(self, d):
            self.encoding = d["encoding"]
            self.payload = d["payload"]
            self.width = d["width"]
            self.height = d["height"]

    for entry in summary.get("masks", []):
        raster = np.frombuffer(decode(_Mask(entry["mask"])), dtype=np.uint8)
        raster = raster.reshape(entry["mask"]["height"], entry["mask"]["width"])
        tile = Image.fromarray((raster * 255).astype(np.uint8), "L")
        bounds = entry["bounds"]
        left, top, right, bottom = (int(round(bounds[k])) for k in ("left", "top", "right", "bottom"))
        if tile.size != (right - left, bottom - top):
            tile = tile.resize((right - left, bottom - top), Image.NEAREST)
        color = Image.new("RGBA", tile.size, (255, 40, 40, 140))
        overlay.paste(color, (left, top), tile)
    Image.alpha_composite(base, overlay).convert("RGB").save(out_path)
    return out_path


def _read_image_data_url(image_path: str) -> str:
    import base64
    import mimetypes

    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def _call_sam_once(client, image_path: str, concept: str,
                         parse_stream: Callable | None = None):
    """Exactly one streaming segmentation request. No retries here by construction."""
    data_url = _read_image_data_url(image_path)
    events = await client.responses.create(
        model=SAM_MODEL,
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        stream=True,
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
    )
    parse = parse_stream or _default_parse_stream
    parsed = parse(events)
    return await parsed.final_result()


def run_sam_experiment(
    *,
    image_path: str,
    concept: str,
    out_path: str,
    experiment_id: str = "sam-mechanical-001",
    ledger_path: str | None = None,
    cache_dir: str | None = None,
    max_cost_usd: float | None = None,
    policy: BudgetPolicy | None = None,
    client_factory: Callable | None = None,
    decode_mask: Callable | None = None,
    parse_stream: Callable | None = None,
) -> ExperimentResult:
    """Run the single-shot SAM segmentation experiment, guarded end to end.

    ``max_cost_usd`` is the operator-declared policy bound for this call;
    ``None`` selects the policy's per-experiment cap. Either way the value is
    enforced by the guard as local policy — it does not cap provider billing.
    """
    policy = policy or BudgetPolicy()
    ledger = UsageLedger(ledger_path or default_ledger_path())
    cache = ResultCache(cache_dir or os.path.join(os.path.dirname(ledger.path), "cache"))
    bound = policy.per_experiment_cap_usd if max_cost_usd is None else max_cost_usd

    if not concept or not concept.strip():
        return ExperimentResult("blocked", "blocked before send: empty concept (invalid input)")
    if not os.path.isfile(image_path):
        return ExperimentResult(
            "blocked", f"blocked before send: image not found: {image_path} (no request sent)")

    asset_hash = sha256_file(image_path)
    cache_key = "|".join([SAM_MODEL, SAM_OPERATION, asset_hash, concept.strip(), "one_bit"])

    cached = cache.get(cache_key)
    if cached and cached.get("v") == CACHE_VERSION:
        render_overlay(image_path, cached, out_path, decode_mask)
        ledger.log(
            provider="meta", model=SAM_MODEL, operation=SAM_OPERATION,
            status="cached", asset_hash=asset_hash, experiment_id=experiment_id,
            image_count=1, cost_status="confirmed", charge_usd=0.0,
            note=f"reused cached result for identical image+concept; no new request sent",
        )
        return ExperimentResult(
            "cached", f"reused cached result for {image_path}; overlay rewritten to {out_path}",
            overlay_path=out_path, from_cache=True,
            box_count=len(cached.get("boxes", [])), mask_count=len(cached.get("masks", [])),
            summary=cached,
        )

    try:
        decision = BudgetGuard(policy, ledger).check(experiment_id, bound)
    except BudgetGuardUnavailable as exc:
        return ExperimentResult(
            "blocked", f"blocked: budget guard unavailable ({exc}); refusing to call")
    if not decision.allowed:
        return ExperimentResult("blocked", f"blocked by budget guard: {decision.reason}")

    try:
        client = (client_factory or _default_client_factory)()
    except MissingCredentialsError as exc:
        ledger.log(
            provider="meta", model=SAM_MODEL, operation=SAM_OPERATION,
            status="blocked", asset_hash=asset_hash, experiment_id=experiment_id,
            image_count=1, cost_status="unknown",
            note=f"blocked before send: {exc} (no request left this machine)",
        )
        return ExperimentResult("blocked", str(exc))

    try:
        result = asyncio.run(_call_sam_once(client, image_path, concept, parse_stream))
    except Exception as exc:  # APIStatusError, ResponsesStreamError, network faults
        ledger.log(
            provider="meta", model=SAM_MODEL, operation=SAM_OPERATION,
            status="error", asset_hash=asset_hash, experiment_id=experiment_id,
            image_count=1, cost_status="unknown",
            note=f"request failed after send: {type(exc).__name__}; "
                 f"cost unknown, may have been billed; no automatic retry",
        )
        return ExperimentResult("error", f"request failed: {type(exc).__name__}: {exc}")

    boxes = [_record_to_dict(r) for r in result.records if r.kind == "box"]
    masks = [_record_to_dict(r) for r in result.records if r.kind == "mask"]
    diagnostics = [
        {"severity": d.severity, "code": d.code, "message": d.message}
        for d in result.diagnostics
    ]
    summary = {
        "v": CACHE_VERSION,
        "model": SAM_MODEL,
        "concept": concept.strip(),
        "asset_hash": asset_hash,
        "boxes": boxes,
        "masks": masks,
        "diagnostics": diagnostics,
        "outcome": getattr(result.outcome, "status", "unknown"),
    }
    if summary["outcome"] != "completed":
        ledger.log(
            provider="meta", model=SAM_MODEL, operation=SAM_OPERATION,
            status="error", asset_hash=asset_hash, experiment_id=experiment_id,
            image_count=1, cost_status="unknown",
            note=f"stream ended early with outcome {summary['outcome']}; "
                 f"cost unknown, may have been billed; no automatic retry",
        )
        return ExperimentResult(
            "error", f"stream ended early: outcome={summary['outcome']}", summary=summary,
            box_count=len(boxes), mask_count=len(masks))

    cache.put(cache_key, summary)
    render_overlay(image_path, summary, out_path, decode_mask)

    usage = _extract_usage(result)
    ledger.log(
        provider="meta", model=SAM_MODEL, operation=SAM_OPERATION,
        status="ok", asset_hash=asset_hash, experiment_id=experiment_id,
        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
        image_count=1,
        charge_usd=usage.get("charge_usd"),
        cost_status="confirmed" if "charge_usd" in usage else "unknown",
        cost_basis="provider-reported charge" if "charge_usd" in usage else (
            "provider returned no charge or token usage; "
            "reconcile against provider dashboard"),
        note=f"{len(boxes)} box(es), {len(masks)} mask(s); overlay at {out_path}",
    )
    return ExperimentResult(
        "ok", f"segmented {image_path}: {len(boxes)} box(es), {len(masks)} mask(s)",
        overlay_path=out_path, box_count=len(boxes), mask_count=len(masks),
        summary=summary)
