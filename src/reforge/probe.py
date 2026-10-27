"""Offline evidence-to-request dry-run orchestration.

Connects the existing local stages into one inspectable workflow without
transmitting anything:

image file → mask-evidence extraction → Spark request construction →
wire-body assembly → guard dry-run → report.

Every stage reuses existing code; this module only pipes data between
stages and renders the deterministic report. It never touches the network,
never reads credentials, never enables a transport, and never writes to
the ledger or cache. Image bytes stay transient (report carries hashes,
sizes, and counts — never base64).
"""

from __future__ import annotations

import json
import mimetypes
import os
from typing import Any, Callable, Mapping

from .baseten_transport import (
    BASETEN_CACHE_VERSION,
    BASETEN_MAX_TOKENS_DEFAULT,
    BASETEN_MODEL_DEFAULT,
    COST_NOTE,
    DATA_URL_UNVERIFIED_NOTE,
    build_baseten_body,
)
from .budget import BudgetPolicy
from .mask_evidence import MaskEvidenceError, extract_mask_evidence
from .spark_request import (
    SPARK_CACHE_VERSION,
    SPARK_MODEL_DEFAULT,
    SparkRequestError,
    build_responses_body,
    build_spark_request,
    dry_run,
)
from .usage_ledger import sha256_file


class ProbeError(ValueError):
    """A safe, explicit dry-run failure (bad image, evidence mismatch, or
    unparsable mask summary). Nothing was sent; nothing was billed."""


def _image_identity(image_path: str) -> dict:
    if not os.path.isfile(image_path):
        raise ProbeError(f"image not found: {os.path.basename(image_path)}")
    try:
        from PIL import Image

        with Image.open(image_path) as handle:
            width, height = int(handle.width), int(handle.height)
    except Exception as exc:
        raise ProbeError(
            f"cannot read image {os.path.basename(image_path)}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ProbeError(f"image has non-positive dimensions: {width}x{height}")
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as handle:
        byte_size = len(handle.read())
    return {"file": os.path.basename(image_path),
            "sha256": sha256_file(image_path),
            "width_px": width, "height_px": height,
            "mime": mime, "bytes": byte_size}


def _load_evidence(mask_summary_path: str | None, image: dict,
                   decode_fn: Callable | None = None) -> tuple[Any | None, dict]:
    """Return (mask evidence record or None, evidence report section)."""
    if mask_summary_path is None:
        return None, {"present": False,
                      "gap": "no mask summary supplied; the request carries "
                             "no mask evidence and image bounds stay unknown"}
    if not os.path.isfile(mask_summary_path):
        raise ProbeError(
            f"mask summary not found: {os.path.basename(mask_summary_path)}")
    try:
        with open(mask_summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ProbeError(f"mask summary is not valid JSON: {exc}") from exc
    try:
        evidence = extract_mask_evidence(
            summary, (image["width_px"], image["height_px"]),
            decode_fn=decode_fn)
    except MaskEvidenceError as exc:
        raise ProbeError(f"mask evidence unusable: {exc}") from exc
    summary_hash = summary.get("asset_hash") if isinstance(summary, Mapping) else None
    if summary_hash and summary_hash != image["sha256"]:
        raise ProbeError(
            "mask summary asset hash does not match the input image; "
            "refusing to attach unrelated evidence")
    return evidence, {
        "present": True,
        "source": os.path.basename(mask_summary_path),
        "asset_hash_match": True,
        "bounds_px": list(evidence.bounds_px),
        "fill_fraction": round(evidence.fill_fraction, 4),
        "boundary_px": evidence.boundary_pixel_count,
        "interior_gap_count": evidence.interior_gap_count,
        "gap_note": ("no detectable interior gaps (not proof of no holes)"
                     if evidence.interior_gap_count == 0
                     else "interior gaps present (image-space only)"),
        "box_confidence": evidence.box_confidence,
        "warnings": list(evidence.warnings),
    }


def prepare_probe(
    image_path: str,
    mask_summary_path: str | None = None,
    *,
    concept: str = "flat metal plate with circular holes",
    classification: str = "rectangular_plate",
    model: str = SPARK_MODEL_DEFAULT,
    max_cost_usd: float | None = None,
    ledger_path: str | None = None,
    policy: BudgetPolicy | None = None,
    decode_fn: Callable | None = None,
) -> dict:
    """Run the full offline dry-run and return a JSON-serializable report.

    Always includes ``transmitted: False``. Raises :class:`ProbeError` for
    unsafe or unusable inputs before any request object exists.
    """
    image = _image_identity(image_path)
    evidence, evidence_report = _load_evidence(mask_summary_path, image,
                                               decode_fn=decode_fn)
    if model is None:
        model = SPARK_MODEL_DEFAULT
    try:
        request = build_spark_request(
            image_path, evidence, classification=classification,
            concept=concept, model=model, max_cost_usd=max_cost_usd)
    except SparkRequestError as exc:
        raise ProbeError(f"request construction failed: {exc}") from exc
    try:
        body = build_responses_body(request)
    except SparkRequestError as exc:
        raise ProbeError(f"wire assembly failed: {exc}") from exc
    wire = {"fields": sorted(body.keys()),
            "content_types": [b["type"] for b in body["input"][0]["content"]],
            "format_type": body["text"]["format"]["type"],
            "format_strict": body["text"]["format"]["strict"],
            "image_data_url_chars": len(body["input"][0]["content"][1]["image_url"])}
    guard_report = dry_run(request, policy=policy, ledger_path=ledger_path)
    return {
        "transmitted": False,
        "note": "offline dry run: no request was sent, nothing was billed",
        "image": image,
        "evidence": evidence_report,
        "request": {
            "model": request.model,
            "classification": request.classification,
            "concept": request.concept,
            "instruction_chars": len(request.instructions),
            "structured_output": guard_report.get("structured_output"),
            "cache_version": SPARK_CACHE_VERSION,
        },
        "wire": wire,
        "guard": {
            "verdict": guard_report["verdict"],
            "reason": guard_report["reason"],
            "bound_usd": guard_report["bound_usd"],
            "discretionary_usd": round(guard_report["discretionary_usd"], 2),
        },
        "transport": "disabled (no live transport exists in this command)",
        "ledger": "unchanged (dry-run performs no ledger writes)",
    }


def prepare_baseten_probe(
    image_path: str,
    mask_summary_path: str | None = None,
    *,
    concept: str = "flat metal plate with circular holes",
    classification: str = "rectangular_plate",
    model: str | None = None,
    max_tokens: int = BASETEN_MAX_TOKENS_DEFAULT,
    max_cost_usd: float | None = None,
    ledger_path: str | None = None,
    policy: BudgetPolicy | None = None,
    decode_fn: Callable | None = None,
) -> dict:
    """Offline Baseten Chat Completions dry-run. Always ``transmitted: False``.

    Reuses the Meta-path request builder for the geometry instructions, then
    assembles the Baseten wire body WITHOUT image bytes (image delivery is
    unapproved and its data-URL compatibility unverified, so dry-run never
    includes image content). Reads the real ledger for totals only, writes
    nothing, touches no network and no credentials.
    """
    from .baseten_transport import BASETEN_BASE_URL_DEFAULT, BASETEN_CHAT_PATH

    image = _image_identity(image_path)
    evidence, evidence_report = _load_evidence(mask_summary_path, image,
                                               decode_fn=decode_fn)
    request = _baseten_instructions(
        image_path, evidence, concept=concept, classification=classification,
        model=model or BASETEN_MODEL_DEFAULT, max_cost_usd=max_cost_usd)
    try:
        body = build_baseten_body(
            model=request.model, instructions=request.instructions,
            max_tokens=max_tokens)
    except SparkRequestError as exc:
        raise ProbeError(f"wire assembly failed: {exc}") from exc
    guard_report = dry_run(request, policy=policy, ledger_path=ledger_path)
    content_types = [part.get("type") for part in
                     body["messages"][0]["content"]]
    return {
        "transmitted": False,
        "note": "offline dry run: no request was sent, nothing was billed",
        "provider": "baseten",
        "endpoint": BASETEN_BASE_URL_DEFAULT + BASETEN_CHAT_PATH,
        "image": image,
        "evidence": evidence_report,
        "request": {
            "model": request.model,
            "classification": request.classification,
            "concept": request.concept,
            "instruction_chars": len(request.instructions),
            "structured_output": "response_format json_object (JSON mode; "
                                 "local validation stays authoritative)",
            "max_tokens": body["max_tokens"],
            "max_tokens_note": COST_NOTE,
            "cache_version": BASETEN_CACHE_VERSION,
            "cache_note": "no Baseten response cache exists in this "
                          "increment; every live call would ledger",
        },
        "wire": {
            "fields": sorted(body.keys()),
            "message_content_types": content_types,
            "response_format": body["response_format"],
            "has_image": False,
            "image_delivery": "none (data-URL compatibility unverified; "
                              "unapproved modes fail closed before sending)",
        },
        "guard": {
            "verdict": guard_report["verdict"],
            "reason": guard_report["reason"],
            "bound_usd": guard_report["bound_usd"],
            "discretionary_usd": round(guard_report["discretionary_usd"], 2),
        },
        "transport": "disabled (no live transport exists in this command)",
        "ledger": "unchanged (dry-run performs no ledger writes)",
        "unresolved": [
            "model slug on the operator account",
            "per-token price and credit applicability",
            DATA_URL_UNVERIFIED_NOTE,
        ],
    }


def _baseten_instructions(image_path: str, evidence: Any, *, concept: str,
                          classification: str, model: str,
                          max_cost_usd: float | None):
    """Build geometry instructions through the established request path."""
    try:
        return build_spark_request(
            image_path, evidence, classification=classification,
            concept=concept, model=model, max_cost_usd=max_cost_usd)
    except SparkRequestError as exc:
        raise ProbeError(f"request construction failed: {exc}") from exc
