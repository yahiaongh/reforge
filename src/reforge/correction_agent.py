"""Bounded correction agent: repairs validation failures deterministically.

Contract with the rest of the pipeline:

- The agent reads an independent :class:`ValidationReport` and the parameter
  set the STL was built from, plus the *user constraints* (known measurements
  that must never be weakened).
- Only parameters NOT marked as user-known (``adjustable``) may be changed,
  and only to the named fields ``length | width | thickness | holes``.
  User-known values are immutable: a failure that can only be fixed by moving
  them is reported as unsolvable, never papered over.
- Deterministic strategies run first. A model is consulted only when they
  yield nothing, only when ``model_diagnose`` is provided, only within
  ``max_correction_calls``, and only when the budget guard allows it.
- Every attempt (including rejected and blocked ones) is recorded. The loop
  stops on success, on no-improvement, or when ``max_attempts`` is reached.
- The agent never generates source code and never executes commands; it
  returns candidate :class:`PlateParams`, and the caller rebuilds through the
  normal validated generate -> render -> validate path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .parameters import Hole, ParameterError, PlateParams, validate_plate_params
from .validator import ValidationReport


#: Fields the agent is allowed to touch. Anything else is rejected even if a
#: (model-proposed) correction asks for it.
ADJUSTABLE_FIELDS = frozenset({"length", "width", "thickness", "holes"})


@dataclass
class CorrectionAttempt:
    index: int
    trigger: str
    change: str
    accepted: bool
    passed: bool
    detail: str


@dataclass
class CorrectionResult:
    success: bool
    params: PlateParams
    report: ValidationReport
    attempts: list[CorrectionAttempt] = field(default_factory=list)
    model_calls_made: int = 0
    detail: str = ""


def failed_check_names(report: ValidationReport) -> list[str]:
    return [c.name for c in report.checks if not c.passed]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _repair_holes_deterministically(params: PlateParams) -> PlateParams | None:
    """Nudge holes into validity (inside plate, non-overlapping).

    Returns a new validated PlateParams, or None when the current holes are
    already valid (nothing to repair) — callers must not count that as an
    attempt.
    """
    holes = [Hole(x=h.x, y=h.y, diameter=h.diameter) for h in params.holes]
    changed = False

    for hole in holes:
        radius = hole.diameter / 2.0
        if hole.diameter > min(params.length, params.width):
            return None  # diameter itself is the problem; not a position fix
        nx = _clamp(hole.x, radius, params.length - radius)
        ny = _clamp(hole.y, radius, params.width - radius)
        if nx != hole.x or ny != hole.y:
            hole.x, hole.y = nx, ny
            changed = True

    # Separate overlaps pairwise along the centres' axis, up to 50 passes.
    for _ in range(50):
        moved = False
        for i in range(len(holes)):
            for j in range(i + 1, len(holes)):
                ri, rj = holes[i].diameter / 2.0, holes[j].diameter / 2.0
                dx, dy = holes[j].x - holes[i].x, holes[j].y - holes[i].y
                dist = (dx * dx + dy * dy) ** 0.5
                need = ri + rj
                if dist < need - 1e-9:
                    push = (need - dist) / 2.0 + 1e-6
                    if dist < 1e-12:
                        dx, dy, dist = 1.0, 0.0, 1.0
                    ux, uy = dx / dist, dy / dist
                    holes[i].x = _clamp(holes[i].x - ux * push, ri, params.length - ri)
                    holes[i].y = _clamp(holes[i].y - uy * push, ri, params.width - ri)
                    holes[j].x = _clamp(holes[j].x + ux * push, rj, params.length - rj)
                    holes[j].y = _clamp(holes[j].y + uy * push, rj, params.width - rj)
                    moved = True
        if moved:
            changed = True
        else:
            break

    if not changed:
        return None
    return validate_plate_params(
        length=params.length,
        width=params.width,
        thickness=params.thickness,
        holes=[(h.x, h.y, h.diameter) for h in holes],
    )


def _reconcile_to_expected(
    params: PlateParams, report: ValidationReport, adjustable: set
) -> PlateParams | None:
    """Set adjustable fields to the failed checks' expected values.

    The independent report carries the original constraints in each failed
    check's ``expected`` slot. Adopting those values for *inferred* fields is
    a deterministic repair; user-known fields are never touched here because
    the caller only passes adjustable fields. Returns None when there is
    nothing applicable to reconcile. May raise ParameterError if the expected
    values are themselves inconsistent.
    """
    axis_field = {"bbox_axis_X": "length", "bbox_axis_Y": "width", "bbox_axis_Z": "thickness"}
    values: dict = {
        "length": params.length,
        "width": params.width,
        "thickness": params.thickness,
        "holes": [(h.x, h.y, h.diameter) for h in params.holes],
    }
    changed = False
    for check in report.checks:
        if check.passed or check.expected is None:
            continue
        if check.name in axis_field and axis_field[check.name] in adjustable:
            values[axis_field[check.name]] = float(check.expected)
            changed = True
        elif check.name.startswith("hole_") and "holes" in adjustable:
            try:
                index = int(check.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if not isinstance(check.expected, dict):
                continue
            if index >= len(values["holes"]):
                return None
            exp = check.expected
            diameter = exp.get("diameter", exp.get("d"))
            values["holes"][index] = (float(exp["x"]), float(exp["y"]), float(diameter))
            changed = True
    if not changed:
        return None
    reconciled = validate_plate_params(
        length=values["length"], width=values["width"],
        thickness=values["thickness"], holes=values["holes"])
    if reconciled == params:
        return None  # expected values already hold; nothing to repair
    return reconciled


class CorrectionAgent:
    def __init__(
        self,
        rebuild: Callable[[PlateParams], ValidationReport],
        max_attempts: int = 2,
        max_correction_calls: int = 2,
        model_diagnose: Callable | None = None,
        budget_guard=None,
        experiment_id: str = "correction",
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.rebuild = rebuild
        self.max_attempts = max_attempts
        self.max_correction_calls = max_correction_calls
        self.model_diagnose = model_diagnose
        self.budget_guard = budget_guard
        self.experiment_id = experiment_id

    def correct(
        self,
        params: PlateParams,
        report: ValidationReport,
        adjustable: frozenset | set = frozenset(),
    ) -> CorrectionResult:
        """Attempt bounded correction of *params* against a failing *report*.

        ``adjustable`` names the top-level fields the agent may change
        (subset of ``length | width | thickness | holes``). Everything else is
        treated as user-known and immutable.
        """
        unknown = set(adjustable) - ADJUSTABLE_FIELDS
        if unknown:
            raise ValueError(f"correction may not adjust {sorted(unknown)}")
        adjustable = set(adjustable)

        attempts: list[CorrectionAttempt] = []
        current_params, current_report = params, report
        model_calls = 0

        if current_report.passed:
            return CorrectionResult(True, current_params, current_report, [],
                                    detail="already valid; no correction needed")

        previous_failures = set(failed_check_names(current_report))

        for index in range(1, self.max_attempts + 1):
            candidate: PlateParams | None = None
            change = ""
            trigger = ",".join(sorted(previous_failures)) or "validation failed"

            # Strategy 1 (deterministic): first repair invalid hole geometry,
            # then reconcile adjustable fields toward the report's expected
            # (constraint) values. ParameterError is recorded, not raised.
            try:
                if "holes" in adjustable:
                    repaired = _repair_holes_deterministically(current_params)
                    if repaired is not None:
                        candidate = repaired
                        change = "moved inferred hole(s) to nearest valid position"
                if candidate is None:
                    reconciled = _reconcile_to_expected(current_params, current_report, adjustable)
                    if reconciled is not None:
                        candidate = reconciled
                        change = "set inferred field(s) to validation-expected values"
            except ParameterError as exc:
                attempts.append(CorrectionAttempt(
                    index, trigger, "deterministic repair",
                    False, False, f"rejected: repair still invalid: {exc}"))

            # Strategy 2 (model): only when deterministic repair yields
            # nothing, a diagnose hook exists, and budget allows.
            if candidate is None and self.model_diagnose is not None:
                if model_calls >= self.max_correction_calls:
                    attempts.append(CorrectionAttempt(
                        index, trigger, "model diagnosis", False, False,
                        f"blocked: correction-call limit ({self.max_correction_calls}) reached"))
                    break
                if self.budget_guard is not None:
                    # Model calls need an explicit pre-call bound: hooks must
                    # expose ``max_cost_usd`` or the call is blocked.
                    bound = getattr(self.model_diagnose, "max_cost_usd", None)
                    decision = self.budget_guard.check(self.experiment_id, bound)
                    if not decision.allowed:
                        attempts.append(CorrectionAttempt(
                            index, trigger, "model diagnosis", False, False,
                            f"blocked by budget guard: {decision.reason}"))
                        break
                try:
                    proposal = self.model_diagnose(
                        failures=sorted(previous_failures),
                        params=current_params,
                        adjustable=sorted(adjustable),
                    )
                except Exception as exc:  # noqa: BLE001 - record, don't crash demo
                    attempts.append(CorrectionAttempt(
                        index, trigger, "model diagnosis", False, False,
                        f"diagnose hook failed: {exc}"))
                    break
                model_calls += 1
                candidate, change = self._apply_proposal(
                    proposal, current_params, adjustable, attempts, index, trigger)
                if candidate is None:
                    continue  # rejection already recorded; try next attempt

            if candidate is None:
                attempts.append(CorrectionAttempt(
                    index, trigger, "no applicable strategy", False, False,
                    "no deterministic repair available and no model call permitted; "
                    "user-known values will not be weakened"))
                break

            # Validate the proposal through the real rebuild path.
            try:
                new_report = self.rebuild(candidate)
            except ParameterError as exc:
                attempts.append(CorrectionAttempt(
                    index, trigger, change, False, False,
                    f"rejected: proposed parameters invalid: {exc}"))
                continue
            new_failures = set(failed_check_names(new_report))
            improved = new_failures < previous_failures
            passed = new_report.passed
            attempts.append(CorrectionAttempt(
                index, trigger, change, True, passed,
                f"failed checks went from {sorted(previous_failures)} "
                f"to {sorted(new_failures)}"))
            current_params, current_report = candidate, new_report
            if passed:
                return CorrectionResult(True, current_params, current_report,
                                        attempts, model_calls,
                                        "correction succeeded; original constraints intact")
            if not improved:
                break  # no progress: stop rather than thrash
            previous_failures = new_failures

        return CorrectionResult(False, current_params, current_report, attempts,
                                model_calls,
                                f"correction unsuccessful after {len(attempts)} attempt(s); "
                                f"user-known values unchanged")

    def _apply_proposal(self, proposal, current_params, adjustable, attempts,
                        index, trigger) -> tuple[PlateParams | None, str]:
        """Apply a model-proposed change dict to adjustable fields only."""
        if not isinstance(proposal, dict):
            attempts.append(CorrectionAttempt(
                index, trigger, "model diagnosis", False, False,
                f"rejected: proposal is {type(proposal).__name__}, not a mapping"))
            return None, ""
        forbidden = [k for k in proposal if k not in adjustable]
        if forbidden:
            attempts.append(CorrectionAttempt(
                index, trigger, "model diagnosis", False, False,
                f"rejected: proposal touches user-known fields {forbidden}"))
            return None, ""
        values: dict = {
            "length": current_params.length,
            "width": current_params.width,
            "thickness": current_params.thickness,
            "holes": [(h.x, h.y, h.diameter) for h in current_params.holes],
        }
        values.update(proposal)
        try:
            candidate = validate_plate_params(
                length=values["length"], width=values["width"],
                thickness=values["thickness"], holes=values["holes"])
        except ParameterError as exc:
            attempts.append(CorrectionAttempt(
                index, trigger, "model diagnosis", False, False,
                f"rejected: proposed parameters invalid: {exc}"))
            return None, ""
        return candidate, f"model-proposed change to {sorted(proposal)}"
