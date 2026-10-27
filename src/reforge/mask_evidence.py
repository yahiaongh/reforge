"""Image-space evidence extraction from cached SAM segmentation masks.

Reads a cached mask summary (as produced by :mod:`sam_experiment`, schema
``v: 1``) plus the source image size, and derives strictly image-space
figures: bounds, fill fraction, boundary-pixel count, and interior-gap
analysis. Offline, deterministic, stdlib + PIL + numpy only.

Hard limits, stated plainly:

- All figures are in **pixels**. No millimetres, no scale, no thickness, no
  hole positions or diameters are produced here.
- Zero detected interior gaps means only that: the mask contains no
  detectable interior gaps. It does **not** establish that the physical
  object has no holes. Physical solidity can never be inferred from a
  segmentation mask.
- Empty, full-frame, and degenerate masks are classified explicitly with
  warnings and never treated as valid physical geometry.
- The ``to_proposal_evidence`` mapping targets the ``evidence`` field of
  :mod:`geometry` ``GeometryProposal`` exactly; any schema conflict raises
  instead of silently reshaping the contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

CACHE_SCHEMA_VERSION = 1

#: Gap-size list cap; the total count is always reported even when capped.
MAX_GAP_SIZES = 25

#: Schema cap mirrored from geometry.MAX_STRING_CHARS so generated notes
#: always fit the evidence field.
MAX_NOTES_CHARS = 500


class MaskEvidenceError(ValueError):
    """Raised for malformed summaries or undecodable mask payloads."""


@dataclass(frozen=True)
class MaskEvidence:
    image_width_px: int
    image_height_px: int
    bounds_px: tuple[float, float, float, float]  # left, top, right, bottom
    raster_width_px: int
    raster_height_px: int
    fill_fraction: float
    boundary_pixel_count: int
    interior_gap_count: int
    interior_gap_sizes_px: tuple[int, ...]
    quant_px_per_cell: tuple[float, float]  # x, y source px per raster cell
    asset_hash: str
    model: str
    concept: str
    box_confidence: float | None
    mask_count: int
    mask_index: int
    warnings: tuple[str, ...]


def _default_decode(mask) -> bytes:
    from meta_sam_parser import decode_mask_to_raster

    return decode_mask_to_raster(mask)


def _require_summary_shape(summary: Any) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        raise MaskEvidenceError(
            f"mask summary must be a mapping, got {type(summary).__name__}")
    if summary.get("v") != CACHE_SCHEMA_VERSION:
        raise MaskEvidenceError(
            f"unsupported mask summary version {summary.get('v')!r}; "
            f"this extractor reads v={CACHE_SCHEMA_VERSION} and refuses to guess")
    masks = summary.get("masks")
    if not isinstance(masks, list) or not masks:
        raise MaskEvidenceError("mask summary contains no masks to analyze")
    return summary


def _eroded(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _interior_gaps(foreground: np.ndarray) -> list[int]:
    """Sizes of background components fully enclosed by foreground.

    Background reaching the raster border is exterior and ignored. Component
    sizes are returned in pixels; interpretation stops there.
    """
    background = ~foreground
    height, width = foreground.shape
    reached = np.zeros_like(background, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if background[y, x]:
                reached[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if background[y, x] and not reached[y, x]:
                reached[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        if y > 0 and background[y - 1, x] and not reached[y - 1, x]:
            reached[y - 1, x] = True
            queue.append((y - 1, x))
        if y < height - 1 and background[y + 1, x] and not reached[y + 1, x]:
            reached[y + 1, x] = True
            queue.append((y + 1, x))
        if x > 0 and background[y, x - 1] and not reached[y, x - 1]:
            reached[y, x - 1] = True
            queue.append((y, x - 1))
        if x < width - 1 and background[y, x + 1] and not reached[y, x + 1]:
            reached[y, x + 1] = True
            queue.append((y, x + 1))
    interior = background & ~reached
    labels = np.zeros((height, width), dtype=np.int32)
    sizes: list[int] = []
    component = 0
    for y in range(height):
        for x in range(width):
            if interior[y, x] and not labels[y, x]:
                component += 1
                size = 0
                queue.append((y, x))
                labels[y, x] = component
                while queue:
                    yy, xx = queue.popleft()
                    size += 1
                    if yy > 0 and interior[yy - 1, xx] and not labels[yy - 1, xx]:
                        labels[yy - 1, xx] = component
                        queue.append((yy - 1, xx))
                    if yy < height - 1 and interior[yy + 1, xx] and not labels[yy + 1, xx]:
                        labels[yy + 1, xx] = component
                        queue.append((yy + 1, xx))
                    if xx > 0 and interior[yy, xx - 1] and not labels[yy, xx - 1]:
                        labels[yy, xx - 1] = component
                        queue.append((yy, xx - 1))
                    if xx < width - 1 and interior[yy, xx + 1] and not labels[yy, xx + 1]:
                        labels[yy, xx + 1] = component
                        queue.append((yy, xx + 1))
                sizes.append(size)
    return sorted(sizes, reverse=True)


def extract_mask_evidence(
    summary: Mapping[str, Any],
    image_size: tuple[int, int],
    decode_fn: Callable[[Any], bytes] | None = None,
) -> MaskEvidence:
    """Analyze the first mask of a cached summary against the image size.

    ``image_size`` is ``(width, height)`` in source pixels. Raises
    :class:`MaskEvidenceError` for malformed input; degenerate masks yield a
    record carrying explicit warnings instead.
    """
    summary = _require_summary_shape(summary)
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or not all(isinstance(v, int) and v > 0 for v in image_size)
    ):
        raise MaskEvidenceError(
            f"image_size must be (width, height) of positive ints, got {image_size!r}")
    image_width, image_height = image_size

    masks = summary["masks"]
    if len(masks) > 1:
        multi_warning = (
            f"summary holds {len(masks)} masks; analyzing masks[0] only")
    else:
        multi_warning = ""
    entry = masks[0]
    if not isinstance(entry, Mapping):
        raise MaskEvidenceError("masks[0] must be a mapping")
    for key in ("bounds", "mask"):
        if key not in entry:
            raise MaskEvidenceError(f"masks[0] is missing {key!r}")
    bounds = entry["bounds"]
    if (
        not isinstance(bounds, Mapping)
        or set(bounds) != {"left", "top", "right", "bottom"}
        or not all(isinstance(bounds[k], (int, float)) for k in bounds)
    ):
        raise MaskEvidenceError("masks[0].bounds must map left/top/right/bottom to numbers")
    left, top, right, bottom = (float(bounds[k]) for k in ("left", "top", "right", "bottom"))
    if not (right > left and bottom > top):
        raise MaskEvidenceError(
            f"masks[0].bounds are degenerate: {(left, top, right, bottom)}")

    payload = entry["mask"]
    if not isinstance(payload, Mapping):
        raise MaskEvidenceError("masks[0].mask must be a mapping")
    for key in ("encoding", "payload", "width", "height"):
        if key not in payload:
            raise MaskEvidenceError(f"masks[0].mask is missing {key!r}")
    raster_width, raster_height = payload["width"], payload["height"]
    if (
        not isinstance(raster_width, int) or not isinstance(raster_height, int)
        or raster_width <= 0 or raster_height <= 0
    ):
        raise MaskEvidenceError(
            f"raster dimensions must be positive ints, got "
            f"{raster_width!r} x {raster_height!r}")

    decode = decode_fn or _default_decode
    try:
        raw = decode(SimpleNamespace(
            encoding=payload["encoding"], payload=payload["payload"],
            width=raster_width, height=raster_height))
        raster = np.frombuffer(raw, dtype=np.uint8).reshape(raster_height, raster_width)
    except Exception as exc:
        raise MaskEvidenceError(f"mask payload failed to decode: {exc}") from exc
    if set(np.unique(raster)) - {0, 1}:
        raise MaskEvidenceError("decoded raster is not binary (values other than 0/1)")
    foreground = raster.astype(bool)

    total = raster.size
    fill = float(foreground.sum()) / total
    boundary = int((foreground & ~_eroded(foreground)).sum())
    gaps = _interior_gaps(foreground)

    warnings: list[str] = []
    if multi_warning:
        warnings.append(multi_warning)
    if left < 0 or top < 0 or right > image_width or bottom > image_height:
        warnings.append("bounds_exceed_image: mask bounds extend past the image edge; "
                        "SAM rounding is suspected, figures still computed")
    if fill == 0.0:
        warnings.append("empty_mask: decoded raster is entirely background; "
                        "there is no segment to measure")
    if (
        left <= 0 and top <= 0 and right >= image_width and bottom >= image_height
        and fill >= 0.99
    ):
        warnings.append("full_frame_mask: mask covers the whole image; it carries "
                        "no object boundary and must not be treated as part geometry")
    if raster_width < 3 or raster_height < 3:
        warnings.append("degenerate_geometry_tiny: raster is smaller than 3x3; "
                        "contour figures are not meaningful")
    if gaps:
        warnings.append(f"interior_gaps_detected: {len(gaps)} interior background "
                        f"region(s) found; image-space only, not confirmed holes")
    else:
        warnings.append("no_detectable_interior_gaps: the mask contains no detectable "
                        "interior gaps; this does not establish that the physical "
                        "object has no holes")

    span_x = right - left
    span_y = bottom - top
    quant = (span_x / raster_width, span_y / raster_height)

    box_confidence = entry.get("confidence")
    if box_confidence is not None and not isinstance(box_confidence, (int, float)):
        raise MaskEvidenceError("masks[0].confidence must be a number or omitted")

    return MaskEvidence(
        image_width_px=image_width,
        image_height_px=image_height,
        bounds_px=(left, top, right, bottom),
        raster_width_px=raster_width,
        raster_height_px=raster_height,
        fill_fraction=fill,
        boundary_pixel_count=boundary,
        interior_gap_count=len(gaps),
        interior_gap_sizes_px=tuple(gaps[:MAX_GAP_SIZES]),
        quant_px_per_cell=quant,
        asset_hash=str(summary.get("asset_hash", "")),
        model=str(summary.get("model", "")),
        concept=str(summary.get("concept", "")),
        box_confidence=None if box_confidence is None else float(box_confidence),
        mask_count=len(masks),
        mask_index=0,
        warnings=tuple(warnings),
    )


def to_proposal_evidence(evidence: MaskEvidence) -> dict:
    """Map a record onto the ``evidence`` field of ``GeometryProposal``.

    Only the four schema-allowed keys are emitted, all values already in
    schema-legal form (units stay pixels; the note says so). Raises
    ``MaskEvidenceError`` if the note cannot fit the schema's string cap
    instead of silently truncating the contract.
    """
    from .geometry import EVIDENCE_KEYS  # local import: geometry stays dependency-free

    asset_hash = evidence.asset_hash
    if asset_hash:
        if not (len(asset_hash) == 64
                and all(c in "0123456789abcdef" for c in asset_hash)):
            raise MaskEvidenceError(
                "asset hash is not a 64-character lowercase hex digest and "
                "conflicts with the GeometryProposal evidence schema")
    if evidence.box_confidence is not None and not (
            0.0 <= evidence.box_confidence <= 1.0):
        raise MaskEvidenceError(
            "box confidence is outside [0, 1] and conflicts with the "
            "GeometryProposal evidence schema")

    gap_summary = (
        f"{evidence.interior_gap_count} interior gap(s); largest "
        f"{evidence.interior_gap_sizes_px[0] if evidence.interior_gap_sizes_px else 0}px"
        if evidence.interior_gap_count
        else "no detectable interior gaps (not proof of no holes)"
    )
    notes = (
        f"image-space mask evidence in pixels (not mm): bounds="
        f"{[round(v, 1) for v in evidence.bounds_px]} fill={evidence.fill_fraction:.4f} "
        f"boundary_px={evidence.boundary_pixel_count} gaps: {gap_summary}; "
        f"raster={evidence.raster_width_px}x{evidence.raster_height_px} "
        f"quant~{evidence.quant_px_per_cell[0]:.2f}x{evidence.quant_px_per_cell[1]:.2f}px/cell; "
        f"warnings=[{'; '.join(evidence.warnings)}]"
    )
    if len(notes) > MAX_NOTES_CHARS:
        raise MaskEvidenceError(
            f"evidence note is {len(notes)} chars, exceeding the schema cap of "
            f"{MAX_NOTES_CHARS}; refusing to truncate provenance")
    record = {
        "mask_bounds": list(evidence.bounds_px),
        "box_confidence": evidence.box_confidence,
        "notes": notes,
    }
    if asset_hash:
        record["image_asset_hash"] = asset_hash
    if set(record) - EVIDENCE_KEYS:
        raise MaskEvidenceError(
            f"generated evidence keys {sorted(set(record) - EVIDENCE_KEYS)} conflict "
            f"with the GeometryProposal schema")
    return record
