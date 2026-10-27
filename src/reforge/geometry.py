"""Structured geometry proposals and their deterministic validation.

A proposal is the uncertainty-aware bridge between image interpretation
(model or human) and the trusted parameter model in :mod:`parameters`.
This module is fully offline: stdlib only, no network, no inference.

Three validation phases, kept distinct so failures stay actionable:

- ``structural`` — shapes, required keys, enums, types, size caps, and a
  scan for executable content. A structural failure means "not a proposal".
- ``plausibility`` — numeric ranges, hole containment/overlap, and internal
  consistency (confidence vs uncertainty vs source). A failure here means
  "a proposal, but geometrically self-contradictory".
- ``confirmation`` — required user-confirmation flags. A failure here means
  "plausible, but not yet authorized for CAD conversion".

Nothing is invented: a null ``value`` stays null (allowed only for
thickness), low confidence never upgrades itself, and uncertain geometry is
never promoted to user-confirmed parameters. Conversion to
:class:`PlateParams` is a later increment, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping

from .parameters import MAX_DIMENSION_MM

#: Part families the current CAD generator can represent.
SUPPORTED_CLASSIFICATIONS = frozenset({"rectangular_plate"})

#: Recognized families the current generator cannot represent. Rejected with
#: an actionable message distinct from truly unknown classifications.
RECOGNIZED_UNGENERATABLE = frozenset(
    {"flat_bracket", "flat_flange", "washer", "spacer"}
)

#: The pipeline works in millimetres throughout; nothing else is convertible.
SUPPORTED_UNITS = frozenset({"mm"})

#: Only convention matching :class:`PlateParams` coordinates is convertible.
SUPPORTED_ORIGINS = frozenset({"plate_corner"})
SUPPORTED_AXES = frozenset({("length", "width", "thickness")})

#: How scale was established. Only a user-supplied reference grounds scale
#: without further confirmation.
SCALE_METHODS = frozenset(
    {"user_measured_reference", "ruler_in_image", "assumed", "unknown"}
)
GROUNDED_SCALE_METHODS = frozenset({"user_measured_reference"})

#: Where a measurement value came from.
MEASUREMENT_SOURCES = frozenset(
    {"observed_pixel", "scale_estimate", "model_inferred", "user_confirmed"}
)

#: Through-holes are the only supported hole kind; null means unknown depth.
SUPPORTED_DEPTHS = frozenset({"through"})

#: Only these hole counts are representable without pagination concerns, and
#: this schema version targets parts that have at least one hole.
MIN_HOLES = 1
MAX_HOLES = 25
MAX_LIST_ITEMS = 20
MAX_STRING_CHARS = 500
MAX_CONFIRMATIONS = 60

#: OpenSCAD/CAD primitives that never belong in proposal prose. All include
#: the opening parenthesis so ordinary words ("cube", "scale", "translate")
#: in legitimate prose are unaffected; only code-shaped calls match.
CAD_PRIMITIVE_MARKERS = (
    "cube(", "cylinder(", "sphere(", "translate(", "rotate(", "scale(",
    "difference(", "union(", "intersection(", "hull(", "minkowski(",
    "linear_extrude(", "rotate_extrude(",
)

#: Substrings that never belong in proposal prose. Matched case-sensitively
#: against every free-text leaf; a hit rejects the proposal structurally.
#: This is the single source of truth for prose screening: response parsing
#: reuses it so mappings and JSON strings face identical screening.
EXECUTABLE_MARKERS = (
    "__", "import ", "exec(", "eval(", "os.", "sys.", "subprocess",
    "${", "#!", "def ", "lambda ", "open(",
) + CAD_PRIMITIVE_MARKERS

EVIDENCE_KEYS = frozenset(
    {"image_asset_hash", "mask_bounds", "box_confidence", "notes"}
)

TOP_LEVEL_KEYS = frozenset(
    {
        "classification", "units", "conventions",
        "length", "width", "thickness", "holes",
        "scale", "assumptions", "unknowns",
        "requires_confirmation", "perspective_warning",
    }
)
OPTIONAL_TOP_LEVEL_KEYS = frozenset({"evidence"})
MEASUREMENT_KEYS = frozenset(
    {"value", "confidence", "uncertainty_mm", "source", "requires_confirmation"}
)


@dataclass(frozen=True)
class ProposalIssue:
    """One deterministic, actionable validation failure."""

    phase: str  # structural | plausibility | confirmation
    field: str
    message: str


class ProposalError(ValueError):
    """Raised when a proposal fails validation, carrying every issue found."""

    def __init__(self, issues: list[ProposalIssue]):
        self.issues = list(issues)
        preview = "; ".join(
            f"[{i.phase}] {i.field}: {i.message}" for i in self.issues[:5]
        )
        if len(self.issues) > 5:
            preview += f"; ... and {len(self.issues) - 5} more"
        super().__init__(
            f"invalid geometry proposal ({len(self.issues)} issue(s)): {preview}"
        )


@dataclass(frozen=True)
class Measurement:
    value: float | None
    confidence: float
    uncertainty_mm: float | None
    source: str
    requires_confirmation: bool


@dataclass(frozen=True)
class HoleProposal:
    x: Measurement
    y: Measurement
    diameter: Measurement
    depth: str | None  # "through" or None (unknown)


@dataclass(frozen=True)
class GeometryProposal:
    classification: str
    units: str
    origin: str
    axes: tuple[str, str, str]
    length: Measurement
    width: Measurement
    thickness: Measurement
    holes: tuple[HoleProposal, ...]
    scale_method: str
    scale_detail: str
    assumptions: tuple[str, ...]
    unknowns: tuple[str, ...]
    requires_confirmation: tuple[str, ...]
    perspective_warning: str
    evidence: Mapping[str, Any] | None


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ProposalIssue] = []

    def add(self, phase: str, field: str, message: str) -> None:
        self.issues.append(ProposalIssue(phase, field, message))

    def ok(self) -> bool:
        return not self.issues


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_prose(issues: _Collector, field: str, value: Any, max_chars: int) -> str | None:
    if not isinstance(value, str):
        issues.add("structural", field, f"must be a string, got {type(value).__name__}")
        return None
    if not value.strip():
        issues.add("structural", field, "must not be empty")
        return None
    if len(value) > max_chars:
        issues.add("structural", field, f"exceeds {max_chars} characters")
        return None
    for marker in EXECUTABLE_MARKERS:
        if marker in value:
            issues.add(
                "structural", field,
                f"contains disallowed executable-like content {marker!r}; "
                f"proposals carry data, never code",
            )
            return None
    return value


def _check_string_list(
    issues: _Collector, field: str, value: Any, *, required: bool
) -> list[str] | None:
    if value is None and not required:
        return []
    if not isinstance(value, list):
        issues.add("structural", field, f"must be a list, got {type(value).__name__}")
        return None
    if len(value) > MAX_LIST_ITEMS:
        issues.add("structural", field, f"exceeds {MAX_LIST_ITEMS} items (oversized)")
        return None
    out: list[str] = []
    for index, item in enumerate(value):
        checked = _check_prose(issues, f"{field}[{index}]", item, MAX_STRING_CHARS)
        if checked is not None:
            out.append(checked)
    return out


def _check_measurement(
    issues: _Collector, field: str, value: Any, *, nullable: bool
) -> Measurement | None:
    if not isinstance(value, Mapping):
        issues.add("structural", field, f"must be a mapping, got {type(value).__name__}")
        return None
    unknown = set(value) - MEASUREMENT_KEYS
    if unknown:
        issues.add("structural", field, f"unknown keys {sorted(unknown)}")
        return None
    missing = [k for k in MEASUREMENT_KEYS if k not in value]
    if missing:
        issues.add("structural", field, f"missing keys {missing}")
        return None

    raw_value = value["value"]
    parsed_value: float | None = None
    if raw_value is None:
        if not nullable:
            issues.add("structural", field, "value must not be null; unknown "
                                           "dimensions are allowed only for thickness")
            return None
    elif _is_number(raw_value):
        parsed_value = float(raw_value)
        if not math.isfinite(parsed_value):
            issues.add("structural", field, "value must be finite")
            return None
    else:
        issues.add("structural", field, f"value must be a number or null, "
                                       f"got {type(raw_value).__name__}")
        return None

    confidence = value["confidence"]
    if not _is_number(confidence) or not math.isfinite(float(confidence)):
        issues.add("structural", field, "confidence must be a finite number")
        return None
    confidence = float(confidence)

    uncertainty = value["uncertainty_mm"]
    parsed_uncertainty: float | None = None
    if uncertainty is not None:
        if not _is_number(uncertainty) or not math.isfinite(float(uncertainty)):
            issues.add("structural", field, "uncertainty_mm must be a finite "
                                           "number or null")
            return None
        parsed_uncertainty = float(uncertainty)

    source = value["source"]
    if source not in MEASUREMENT_SOURCES:
        issues.add("structural", field, f"source must be one of "
                                       f"{sorted(MEASUREMENT_SOURCES)}, got {source!r}")
        return None

    requires_confirmation = value["requires_confirmation"]
    if not isinstance(requires_confirmation, bool):
        issues.add("structural", field, "requires_confirmation must be a boolean")
        return None

    if parsed_value is None:
        if confidence != 0.0:
            issues.add("plausibility", field, "null value must carry confidence 0.0; "
                                              "absence of a value is not evidence")
            return None
        if not requires_confirmation:
            issues.add("confirmation", field, "null value must require user confirmation")
            return None
    else:
        if not 0.0 <= confidence <= 1.0:
            issues.add("plausibility", field, f"confidence {confidence} outside [0, 1]")
            return None
        if confidence == 0.0:
            issues.add("plausibility", field, "a stated value with confidence 0.0 is "
                                              "contradictory; use null instead")
            return None
        if parsed_uncertainty is not None and parsed_uncertainty < 0:
            issues.add("plausibility", field, "uncertainty_mm must not be negative")
            return None
        if confidence < 1.0 and not (parsed_uncertainty or 0.0) > 0:
            issues.add("plausibility", field, "confidence below 1.0 requires a "
                                              "positive uncertainty_mm; uncertainty "
                                              "must be quantified, not implied")
            return None
        # Confidence (credence in the value) and uncertainty_mm (error bound)
        # are independent: confidence 1.0 with a positive uncertainty is a
        # legitimate certain-within-tolerance claim (e.g. a user-confirmed
        # caliper reading with instrument precision). Negative uncertainty is
        # already rejected above for every confidence level.
        if source == "user_confirmed" and requires_confirmation:
            issues.add("plausibility", field, "source user_confirmed contradicts "
                                              "requires_confirmation=true")
            return None

    return Measurement(
        value=parsed_value,
        confidence=confidence if parsed_value is not None else 0.0,
        uncertainty_mm=parsed_uncertainty,
        source=source,
        requires_confirmation=requires_confirmation,
    )


def _check_positive_dim(
    issues: _Collector, field: str, measurement: Measurement | None
) -> None:
    if measurement is None or measurement.value is None:
        return
    if measurement.value <= 0:
        issues.add("plausibility", field, f"value {measurement.value} must be positive")
    elif measurement.value > MAX_DIMENSION_MM:
        issues.add("plausibility", field, f"value {measurement.value} exceeds maximum "
                                          f"{MAX_DIMENSION_MM} mm (suspected unit error)")


def validate_proposal(raw: Any) -> GeometryProposal:
    """Validate a parsed proposal mapping, gathering every issue found.

    Raises :class:`ProposalError` listing all issues (each tagged with its
    phase and field) when anything is malformed, inconsistent, or awaiting
    required confirmation. Never invents values or upgrades certainty.
    """
    issues = _Collector()
    if not isinstance(raw, Mapping):
        raise ProposalError([ProposalIssue(
            "structural", "<proposal>",
            f"proposal must be a mapping, got {type(raw).__name__}")])

    unknown = set(raw) - TOP_LEVEL_KEYS - OPTIONAL_TOP_LEVEL_KEYS
    if unknown:
        issues.add("structural", "<proposal>", f"unknown top-level keys {sorted(unknown)}")
    missing = [k for k in TOP_LEVEL_KEYS if k not in raw]
    for key in missing:
        issues.add("structural", key, "missing required field")

    classification = raw.get("classification")
    if classification not in SUPPORTED_CLASSIFICATIONS:
        if classification in RECOGNIZED_UNGENERATABLE:
            issues.add("structural", "classification",
                         f"{classification!r} is recognized but the current CAD "
                         f"generator cannot represent it; only "
                         f"{sorted(SUPPORTED_CLASSIFICATIONS)} are convertible")
        else:
            issues.add("structural", "classification",
                         f"unsupported classification {classification!r}; expected one of "
                         f"{sorted(SUPPORTED_CLASSIFICATIONS)}")

    units = raw.get("units")
    if units not in SUPPORTED_UNITS:
        issues.add("structural", "units", f"unsupported units {units!r}; the pipeline "
                                         f"works in millimetres ({sorted(SUPPORTED_UNITS)})")

    origin: str | None = None
    axes: tuple[str, str, str] | None = None
    conventions = raw.get("conventions")
    if not isinstance(conventions, Mapping):
        issues.add("structural", "conventions",
                     f"must be a mapping, got {type(conventions).__name__}")
    else:
        if set(conventions) - {"origin", "axes"}:
            issues.add("structural", "conventions",
                         f"unknown keys {sorted(set(conventions) - {'origin', 'axes'})}")
        origin = conventions.get("origin")
        if origin not in SUPPORTED_ORIGINS:
            issues.add("structural", "conventions.origin",
                         f"unsupported origin {origin!r}; convertible proposals use "
                         f"{sorted(SUPPORTED_ORIGINS)} (plate corner at (0, 0), "
                         f"x along length, y along width)")
        raw_axes = conventions.get("axes")
        if (
            not isinstance(raw_axes, Mapping)
            or set(raw_axes) != {"x", "y", "z"}
            or tuple(raw_axes[k] for k in ("x", "y", "z")) not in SUPPORTED_AXES
        ):
            issues.add("structural", "conventions.axes",
                         'axes must be exactly {"x": "length", "y": "width", '
                         '"z": "thickness"}')
        else:
            axes = (raw_axes["x"], raw_axes["y"], raw_axes["z"])

    length = _check_measurement(issues, "length", raw.get("length"), nullable=False) \
        if "length" in raw else None
    width = _check_measurement(issues, "width", raw.get("width"), nullable=False) \
        if "width" in raw else None
    thickness = _check_measurement(issues, "thickness", raw.get("thickness"), nullable=True) \
        if "thickness" in raw else None
    _check_positive_dim(issues, "length", length)
    _check_positive_dim(issues, "width", width)
    if thickness is not None and thickness.value is not None:
        _check_positive_dim(issues, "thickness", thickness)

    holes: list[HoleProposal] = []
    raw_holes = raw.get("holes", None) if isinstance(raw, Mapping) else None
    if "holes" in raw:
        if not isinstance(raw_holes, list):
            issues.add("structural", "holes",
                         f"must be a list, got {type(raw_holes).__name__}")
        elif not raw_holes:
            issues.add("structural", "holes", "hole geometry is missing; this schema "
                                              "version requires at least one candidate hole")
        elif len(raw_holes) > MAX_HOLES:
            issues.add("structural", "holes", f"exceeds {MAX_HOLES} holes (oversized)")
        else:
            for index, raw_hole in enumerate(raw_holes):
                prefix = f"holes[{index}]"
                if not isinstance(raw_hole, Mapping):
                    issues.add("structural", prefix,
                                 f"must be a mapping, got {type(raw_hole).__name__}")
                    continue
                if set(raw_hole) - {"x", "y", "diameter", "depth"}:
                    issues.add("structural", prefix,
                                 f"unknown keys {sorted(set(raw_hole) - {'x', 'y', 'diameter', 'depth'})}")
                    continue
                hole_measurements = {}
                valid = True
                for key in ("x", "y", "diameter"):
                    if key not in raw_hole:
                        issues.add("structural", f"{prefix}.{key}", "missing required field")
                        valid = False
                    else:
                        parsed = _check_measurement(issues, f"{prefix}.{key}",
                                                    raw_hole[key], nullable=False)
                        if parsed is None:
                            valid = False
                        else:
                            hole_measurements[key] = parsed
                depth = raw_hole.get("depth")
                if depth is not None and depth not in SUPPORTED_DEPTHS:
                    issues.add("structural", f"{prefix}.depth",
                                 f"unsupported depth {depth!r}; only through-holes "
                                 f"({sorted(SUPPORTED_DEPTHS)}) or null (unknown) are "
                                 f"representable")
                    valid = False
                if valid:
                    holes.append(HoleProposal(
                        x=hole_measurements["x"], y=hole_measurements["y"],
                        diameter=hole_measurements["diameter"], depth=depth))
            for hole in holes:
                if hole.diameter.value is not None:
                    _check_positive_dim(issues, "holes.diameter", hole.diameter)

    # Geometric plausibility across values present (skipped for nulls, which
    # stay unknown rather than being checked against invented numbers).
    plain = [h for h in holes
             if h.x.value is not None and h.y.value is not None
             and h.diameter.value is not None]
    if (length and length.value is not None
            and width and width.value is not None and plain):
        for index, hole in enumerate(plain):
            radius = hole.diameter.value / 2.0
            if hole.diameter.value > min(length.value, width.value):
                issues.add("plausibility", f"holes[{index}].diameter",
                             f"diameter {hole.diameter.value} exceeds smallest plate side")
            if hole.x.value - radius < 0 or hole.x.value + radius > length.value:
                issues.add("plausibility", f"holes[{index}]",
                             f"hole at x={hole.x.value} extends outside plate length "
                             f"{length.value}")
            if hole.y.value - radius < 0 or hole.y.value + radius > width.value:
                issues.add("plausibility", f"holes[{index}]",
                             f"hole at y={hole.y.value} extends outside plate width "
                             f"{width.value}")
        for i in range(len(plain)):
            for j in range(i + 1, len(plain)):
                distance = math.hypot(plain[i].x.value - plain[j].x.value,
                                      plain[i].y.value - plain[j].y.value)
                if distance < (plain[i].diameter.value + plain[j].diameter.value) / 2.0 - 1e-9:
                    issues.add("plausibility", f"holes[{i}]",
                                 f"holes[{i}] and holes[{j}] overlap "
                                 f"(centre distance {distance:.4f} mm)")

    scale_method: str | None = None
    scale = raw.get("scale")
    if not isinstance(scale, Mapping):
        issues.add("structural", "scale", f"must be a mapping, got {type(scale).__name__}")
    else:
        if set(scale) - {"method", "detail"}:
            issues.add("structural", "scale",
                         f"unknown keys {sorted(set(scale) - {'method', 'detail'})}")
        scale_method = scale.get("method")
        if scale_method not in SCALE_METHODS:
            issues.add("structural", "scale.method",
                         f"must be one of {sorted(SCALE_METHODS)}, got {scale_method!r}")
        detail = scale.get("detail")
        if "detail" in scale:
            _check_prose(issues, "scale.detail", detail, MAX_STRING_CHARS)

    assumptions = _check_string_list(issues, "assumptions", raw.get("assumptions"),
                                     required=True) if "assumptions" in raw else None
    unknowns = _check_string_list(issues, "unknowns", raw.get("unknowns"),
                                  required=True) if "unknowns" in raw else None
    perspective = _check_prose(issues, "perspective_warning",
                               raw.get("perspective_warning"), MAX_STRING_CHARS) \
        if "perspective_warning" in raw else None

    confirmations: list[str] = []
    raw_confirmations = raw.get("requires_confirmation")
    if "requires_confirmation" in raw:
        if not isinstance(raw_confirmations, list):
            issues.add("structural", "requires_confirmation",
                         f"must be a list, got {type(raw_confirmations).__name__}")
        elif len(raw_confirmations) > MAX_CONFIRMATIONS:
            issues.add("structural", "requires_confirmation",
                         f"exceeds {MAX_CONFIRMATIONS} entries (oversized)")
        else:
            for entry in raw_confirmations:
                if not isinstance(entry, str):
                    issues.add("structural", "requires_confirmation",
                                 f"entries must be strings, got {type(entry).__name__}")
                else:
                    confirmations.append(entry)

    evidence = None
    if "evidence" in raw:
        evidence = raw["evidence"]
        if not isinstance(evidence, Mapping):
            issues.add("structural", "evidence",
                         f"must be a mapping, got {type(evidence).__name__}")
            evidence = None
        else:
            unknown_evidence = set(evidence) - EVIDENCE_KEYS
            if unknown_evidence:
                issues.add("structural", "evidence",
                             f"unknown keys {sorted(unknown_evidence)}")
                evidence = None
            else:
                asset_hash = evidence.get("image_asset_hash")
                if asset_hash is not None and not (
                        isinstance(asset_hash, str) and len(asset_hash) == 64
                        and all(c in "0123456789abcdef" for c in asset_hash)):
                    issues.add("structural", "evidence.image_asset_hash",
                                 "must be a 64-character lowercase hex digest or omitted")
                bounds = evidence.get("mask_bounds")
                if bounds is not None and not (
                        isinstance(bounds, list) and len(bounds) == 4
                        and all(_is_number(v) and math.isfinite(float(v)) for v in bounds)):
                    issues.add("structural", "evidence.mask_bounds",
                                 "must be a list of 4 finite numbers or omitted")
                box_confidence = evidence.get("box_confidence")
                if box_confidence is not None and not (
                        _is_number(box_confidence)
                        and 0.0 <= float(box_confidence) <= 1.0):
                    issues.add("structural", "evidence.box_confidence",
                                 "must be within [0, 1] or omitted")
                notes = evidence.get("notes")
                if notes is not None:
                    _check_prose(issues, "evidence.notes", notes, MAX_STRING_CHARS)

    if not issues.ok():
        raise ProposalError(issues.issues)

    # Confirmation phase: every flagged measurement must be listed, every
    # listed path must exist, and ungrounded scale / unknown thickness or
    # depth must be listed. Checked only after structural parsing so paths
    # resolve against real fields.
    confirmation = _Collector()
    flagged: set[str] = set()
    for name, measurement in (("length", length), ("width", width),
                              ("thickness", thickness)):
        if measurement is not None and measurement.requires_confirmation:
            flagged.add(name)
    for index, hole in enumerate(holes):
        for key, measurement in (("x", hole.x), ("y", hole.y),
                                 ("diameter", hole.diameter)):
            if measurement.requires_confirmation:
                flagged.add(f"holes[{index}].{key}")
        if hole.depth is None:
            flagged.add(f"holes[{index}]")
    if thickness is not None and thickness.value is None:
        flagged.add("thickness")
    if scale_method is not None and scale_method not in GROUNDED_SCALE_METHODS:
        flagged.add("scale")

    listed = set(confirmations)
    for path in sorted(flagged - listed):
        confirmation.add("confirmation", "requires_confirmation",
                           f"{path} requires user confirmation but is not listed")
    valid_paths = set(flagged)
    for index in range(len(holes)):
        valid_paths.update(f"holes[{index}].{key}" for key in ("x", "y", "diameter"))
        valid_paths.add(f"holes[{index}]")
    valid_paths.update({"length", "width", "thickness", "scale"})
    for path in sorted(listed - valid_paths):
        confirmation.add("confirmation", "requires_confirmation",
                           f"{path} is listed but matches no field of this proposal")
    for path in sorted(listed & valid_paths - flagged):
        confirmation.add("confirmation", "requires_confirmation",
                           f"{path} is listed but not flagged as requiring confirmation; "
                           f"listing must match the field's requires_confirmation state")
    if not confirmation.ok():
        raise ProposalError(confirmation.issues)

    return GeometryProposal(
        classification=classification,
        units=units,
        origin=origin,
        axes=axes,
        length=length,
        width=width,
        thickness=thickness,
        holes=tuple(holes),
        scale_method=scale_method,
        scale_detail=scale.get("detail") if isinstance(scale, Mapping) else "",
        assumptions=tuple(assumptions or []),
        unknowns=tuple(unknowns or []),
        requires_confirmation=tuple(confirmations),
        perspective_warning=perspective,
        evidence=evidence,
    )


def parse_proposal_json(text: str) -> GeometryProposal:
    """Parse strict JSON text into a validated proposal.

    ``parse_constant`` rejects ``NaN``/``Infinity`` literals so non-finite
    doubles can never enter validation, and duplicate handling follows
    ``json.loads`` deterministically (last key wins) before validation.
    """
    if not isinstance(text, str):
        raise ProposalError([ProposalIssue(
            "structural", "<proposal>",
            f"JSON text must be a string, got {type(text).__name__}")])
    try:
        raw = json.loads(
            text,
            parse_constant=lambda name: (_ for _ in ()).throw(
                ValueError(f"non-finite literal {name} is not valid proposal JSON")),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProposalError([ProposalIssue(
            "structural", "<proposal>", f"malformed JSON: {exc}")]) from exc
    return validate_proposal(raw)
