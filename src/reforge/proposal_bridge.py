"""Deterministic bridge from validated geometry proposals to CAD parameters.

Converts a :class:`GeometryProposal` into :class:`PlateParams` only when the
proposal is sufficiently specified and sufficiently grounded. Anything else
is rejected with actionable reasons — never defaulted, never guessed, never
upgraded in confidence.

Conversion contract (all must hold, checked in order):

1. The input validates under :func:`validate_proposal` (mappings are
   validated; already-validated proposals are used as-is — they are frozen).
2. Classification is ``rectangular_plate`` with ``mm`` units and the
   plate-corner convention (guaranteed by validation; re-asserted here so the
   bridge never silently follows a future schema widening).
3. Every dimension consumed (length, width, thickness, each hole x/y/diameter)
   is non-null and carries ``requires_confirmation == False`` — i.e. a human
   (or an authorized upstream step) has already confirmed it. Pending
   confirmation is a rejection, not a default.
4. Scale is grounded: ``scale_method`` is ``user_measured_reference``, OR
   every consumed measurement has source ``user_confirmed`` (directly
   measured numbers need no image scale). Image-derived estimates without a
   grounded scale are rejected, never rescaled by assumption.
5. Every hole declares depth ``"through"``. Unknown depth is incomplete hole
   geometry and is rejected — ``PlateParams`` models through-holes only.
6. Final construction goes through :func:`validate_plate_params`, so hole
   containment, overlap, and dimensional limits are enforced by the existing
   trust boundary, not reimplemented here.

Image-space mask evidence (pixel bounds, fill, gaps) is carried through as
provenance only. It is never converted into millimetres by this module;
scale comes exclusively from condition 4.

The result preserves per-field provenance (value, source, confidence,
confirmation state) alongside the parameters, because ``PlateParams``
itself carries numbers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .geometry import (
    GeometryProposal,
    SUPPORTED_AXES,
    SUPPORTED_CLASSIFICATIONS,
    SUPPORTED_ORIGINS,
    SUPPORTED_UNITS,
    validate_proposal,
)
from .parameters import ParameterError, PlateParams, validate_plate_params


class BridgeError(ValueError):
    """Raised when a valid proposal is not convertible, listing every reason."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__(
            f"proposal not convertible to CAD parameters "
            f"({len(self.reasons)} reason(s)): {'; '.join(self.reasons[:5])}"
            + (f"; ... and {len(self.reasons) - 5} more" if len(self.reasons) > 5 else "")
        )


@dataclass(frozen=True)
class FieldProvenance:
    value: float
    source: str
    confidence: float
    uncertainty_mm: float | None
    requires_confirmation: bool


@dataclass(frozen=True)
class ConversionResult:
    params: PlateParams
    provenance: Mapping[str, FieldProvenance]
    scale_method: str
    evidence_summary: Mapping[str, Any]


def convert_proposal(proposal: GeometryProposal | Mapping[str, Any]) -> ConversionResult:
    """Convert a validated proposal into CAD parameters plus provenance.

    Raises :class:`ProposalError` for invalid input mappings and
    :class:`BridgeError` for valid proposals that are not sufficiently
    specified or grounded for safe CAD generation.
    """
    if not isinstance(proposal, GeometryProposal):
        proposal = validate_proposal(proposal)

    reasons: list[str] = []

    if proposal.classification not in SUPPORTED_CLASSIFICATIONS:
        reasons.append(
            f"classification {proposal.classification!r} is not convertible; "
            f"bridge supports {sorted(SUPPORTED_CLASSIFICATIONS)}")
    if proposal.units not in SUPPORTED_UNITS:
        reasons.append(f"units {proposal.units!r} are not convertible to the mm pipeline")
    if proposal.origin not in SUPPORTED_ORIGINS or proposal.axes not in SUPPORTED_AXES:
        reasons.append(
            f"conventions {proposal.origin!r}/{proposal.axes!r} do not match the "
            f"plate-corner PlateParams coordinate frame")

    values: dict[str, float] = {}
    provenance: dict[str, FieldProvenance] = {}

    def consume(path: str, measurement, *, nullable: bool = False) -> None:
        if measurement.value is None:
            if not nullable:
                reasons.append(f"{path} has no value; unknown dimensions are rejected, "
                               f"never defaulted")
            return
        if measurement.requires_confirmation:
            reasons.append(f"{path} still requires user confirmation; confirmed values only")
            return
        values[path] = float(measurement.value)
        provenance[path] = FieldProvenance(
            value=float(measurement.value),
            source=measurement.source,
            confidence=measurement.confidence,
            uncertainty_mm=measurement.uncertainty_mm,
            requires_confirmation=measurement.requires_confirmation,
        )

    consume("length", proposal.length)
    consume("width", proposal.width)
    consume("thickness", proposal.thickness)
    for index, hole in enumerate(proposal.holes):
        consume(f"holes[{index}].x", hole.x)
        consume(f"holes[{index}].y", hole.y)
        consume(f"holes[{index}].diameter", hole.diameter)
        if hole.depth != "through":
            reasons.append(
                f"holes[{index}] has unknown depth; only through-holes are "
                f"representable and depth must not be guessed")

    all_measurements = [proposal.length, proposal.width, proposal.thickness]
    for hole in proposal.holes:
        all_measurements += [hole.x, hole.y, hole.diameter]
    all_confirmed_direct = all(m.source == "user_confirmed" for m in all_measurements)
    if proposal.scale_method != "user_measured_reference" and not all_confirmed_direct:
        reasons.append(
            f"scale method {proposal.scale_method!r} is not grounded; image-derived "
            f"estimates require a user-measured reference (or fully user-confirmed "
            f"dimensions) before CAD conversion")

    if reasons:
        raise BridgeError(reasons)

    holes = [
        (values[f"holes[{i}].x"], values[f"holes[{i}].y"], values[f"holes[{i}].diameter"])
        for i in range(len(proposal.holes))
    ]
    try:
        params = validate_plate_params(
            length=values["length"], width=values["width"],
            thickness=values["thickness"], holes=holes)
    except ParameterError as exc:
        raise BridgeError([f"converted dimensions fail parameter constraints: {exc}"]) from exc

    evidence_summary: dict[str, Any] = {}
    if proposal.evidence:
        evidence_summary = {
            "image_asset_hash": proposal.evidence.get("image_asset_hash"),
            "mask_bounds_px": proposal.evidence.get("mask_bounds"),
            "box_confidence": proposal.evidence.get("box_confidence"),
            "note": "image-space evidence carried as provenance only; "
                    "never converted to millimetres by this bridge",
        }

    return ConversionResult(
        params=params,
        provenance=provenance,
        scale_method=proposal.scale_method,
        evidence_summary=evidence_summary,
    )
