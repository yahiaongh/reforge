"""Parameter definitions and deterministic validation.

All dimensions are in millimetres. This module never calls a model and never
touches the network: it is the trust boundary between interpreted (possibly
AI-proposed) values and CAD generation.

Unknown dimensions must be represented as missing (None) by the caller, never
as invented exact numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

#: Hard upper bound (mm) guarding against unit mistakes (e.g. metres given as
#: millimetres). Documented, deterministic, and deliberately generous.
MAX_DIMENSION_MM = 10000.0


class ParameterError(ValueError):
    """Raised when a parameter set is missing, malformed, or geometrically
    impossible."""


@dataclass(frozen=True)
class Hole:
    """A circular through-hole. ``x``/``y`` are centre coordinates measured
    from the plate corner (0, 0); ``diameter`` is the hole diameter."""

    x: float
    y: float
    diameter: float


@dataclass(frozen=True)
class PlateParams:
    """Validated parameters for a rectangular plate with circular holes."""

    length: float  # X extent, mm
    width: float  # Y extent, mm
    thickness: float  # Z extent, mm
    holes: tuple[Hole, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "width": self.width,
            "thickness": self.thickness,
            "holes": [
                {"x": h.x, "y": h.y, "diameter": h.diameter} for h in self.holes
            ],
        }


def _require_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"'{name}' must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"'{name}' must be finite, got {value!r}")
    return value


def _require_positive(name: str, value: Any, maximum: float = MAX_DIMENSION_MM) -> float:
    number = _require_number(name, value)
    if number <= 0:
        raise ParameterError(f"'{name}' must be positive, got {number!r}")
    if number > maximum:
        raise ParameterError(f"'{name}' exceeds maximum {maximum} mm, got {number!r}")
    return number


def _parse_hole(index: int, raw: Any, default_diameter: float | None) -> Hole:
    if isinstance(raw, Mapping):
        unknown = set(raw) - {"x", "y", "diameter"}
        if unknown:
            raise ParameterError(f"holes[{index}]: unknown keys {sorted(unknown)}")
        try:
            x = raw["x"]
            y = raw["y"]
        except KeyError as exc:
            raise ParameterError(f"holes[{index}]: missing key {exc}") from exc
        diameter = raw.get("diameter", default_diameter)
        if diameter is None:
            raise ParameterError(f"holes[{index}]: missing 'diameter' and no default given")
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) == 2 and default_diameter is not None:
            x, y = raw
            diameter = default_diameter
        elif len(raw) == 3:
            x, y, diameter = raw
        else:
            raise ParameterError(
                f"holes[{index}]: expected [x, y] (with default diameter) or "
                f"[x, y, diameter], got {list(raw)!r}"
            )
    else:
        raise ParameterError(
            f"holes[{index}]: expected mapping or sequence, got {type(raw).__name__}"
        )
    x = _require_number(f"holes[{index}].x", x)
    y = _require_number(f"holes[{index}].y", y)
    diameter = _require_positive(f"holes[{index}].diameter", diameter)
    return Hole(x=x, y=y, diameter=diameter)


def validate_plate_params(
    length: Any,
    width: Any,
    thickness: Any,
    holes: Sequence[Any] | None = None,
    hole_diameter: Any | None = None,
) -> PlateParams:
    """Validate raw inputs and return an immutable :class:`PlateParams`.

    Raises :class:`ParameterError` for nonpositive dimensions, invalid hole
    diameters, holes outside the part, overlapping holes, or malformed values.
    """
    length = _require_positive("length", length)
    width = _require_positive("width", width)
    thickness = _require_positive("thickness", thickness)

    default_diameter: float | None = None
    if hole_diameter is not None:
        default_diameter = _require_positive("hole_diameter", hole_diameter)

    parsed: list[Hole] = []
    for index, raw in enumerate(holes or []):
        parsed.append(_parse_hole(index, raw, default_diameter))

    for index, hole in enumerate(parsed):
        radius = hole.diameter / 2.0
        if hole.diameter > min(length, width):
            raise ParameterError(
                f"holes[{index}]: diameter {hole.diameter} exceeds smallest plate "
                f"side {min(length, width)}"
            )
        if hole.x - radius < 0 or hole.x + radius > length:
            raise ParameterError(
                f"holes[{index}]: hole at x={hole.x} with diameter {hole.diameter} "
                f"extends outside plate length {length}"
            )
        if hole.y - radius < 0 or hole.y + radius > width:
            raise ParameterError(
                f"holes[{index}]: hole at y={hole.y} with diameter {hole.diameter} "
                f"extends outside plate width {width}"
            )

    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            distance = math.hypot(parsed[i].x - parsed[j].x, parsed[i].y - parsed[j].y)
            if distance < (parsed[i].diameter + parsed[j].diameter) / 2.0 - 1e-9:
                raise ParameterError(
                    f"holes[{i}] and holes[{j}]: holes overlap "
                    f"(centre distance {distance:.4f} mm)"
                )

    return PlateParams(length=length, width=width, thickness=thickness, holes=tuple(parsed))


#: Keys accepted by :func:`plate_params_from_dict`. Anything else is rejected
#: so model-drift keys cannot slip silently into CAD generation.
PLATE_DICT_KEYS = frozenset({"length", "width", "thickness", "holes", "hole_diameter"})


def plate_params_from_dict(raw: Mapping[str, Any]) -> PlateParams:
    """Build validated :class:`PlateParams` from a plain mapping (JSON / model
    output). Missing required keys, unknown keys, and ``None`` values raise
    :class:`ParameterError` — a missing dimension is an error, never silently
    replaced by an estimate."""
    if not isinstance(raw, Mapping):
        raise ParameterError(f"parameter set must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - PLATE_DICT_KEYS
    if unknown:
        raise ParameterError(f"unknown parameter keys: {sorted(unknown)}")
    missing = [k for k in ("length", "width", "thickness") if k not in raw]
    if missing:
        raise ParameterError(f"missing required parameters: {missing}")
    if any(raw[k] is None for k in ("length", "width", "thickness")):
        raise ParameterError("dimensions must not be None; mark unknown dimensions as missing")
    return validate_plate_params(
        length=raw["length"],
        width=raw["width"],
        thickness=raw["thickness"],
        holes=raw.get("holes") or [],
        hole_diameter=raw.get("hole_diameter"),
    )
