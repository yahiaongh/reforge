"""Correction-agent tests: deterministic fixes, limits, and no paid calls.

Real OpenSCAD rebuilds are used for the success path; lightweight fake
rebuilds isolate the agent's decision logic everywhere else. No paid API is
touched: the model hook is always a local mock or None.
"""

import unittest

from tests import TempDirTest, needs_openscad, sample_params
from reforge.budget import BudgetGuard, BudgetPolicy
from reforge.cad_generator import write_scad
from reforge.correction_agent import CorrectionAgent
from reforge.parameters import validate_plate_params
from reforge.renderer import render_scad_to_stl
from reforge.usage_ledger import UsageLedger
from reforge.validator import Check, ValidationReport, validate_stl


def fake_report(failures: dict) -> ValidationReport:
    """Build a report whose failed checks carry ``expected`` constraint values.

    ``failures`` maps check name -> expected value.
    """
    checks = [Check(name, False, f"{name} failed", expected=exp)
              for name, exp in failures.items()]
    checks.append(Check("file_present", True, "fake"))
    return ValidationReport(stl_path="<fake>", passed=not failures, checks=checks)


def passing_report() -> ValidationReport:
    return ValidationReport(stl_path="<fake>", passed=True,
                            checks=[Check("file_present", True, "fake")])


CONSTRAINTS = validate_plate_params(
    length=60, width=40, thickness=5,
    holes=[(10, 10), (50, 10), (10, 30), (50, 30)], hole_diameter=6)


def real_rebuild(case: TempDirTest, constraints):
    """Rebuild through the real local path, validating against constraints."""

    def rebuild(params):
        scad = case.path("correction.scad")
        stl = case.path("correction.stl")
        write_scad(params, scad)
        render_scad_to_stl(scad, stl)
        return validate_stl(stl, constraints)

    return rebuild


class TestDeterministicCorrection(TempDirTest):
    @needs_openscad
    def test_correctable_hole_position_succeeds(self):
        # Valid params, but the inferred hole sits away from the constraint.
        params = validate_plate_params(
            length=60, width=40, thickness=5,
            holes=[(14, 10), (50, 10), (10, 30), (50, 30)], hole_diameter=6)
        agent = CorrectionAgent(rebuild=real_rebuild(self, CONSTRAINTS))
        initial = agent.rebuild(params)
        self.assertFalse(initial.passed)
        result = agent.correct(params, initial, adjustable={"holes"})
        self.assertTrue(result.success, msg=str([a.detail for a in result.attempts]))
        self.assertEqual(result.model_calls_made, 0)  # deterministic: no paid call
        self.assertLessEqual(len(result.attempts), 2)
        # Original constraints hold in the final parameters.
        self.assertEqual(result.params.to_dict(), CONSTRAINTS.to_dict())

    def test_already_valid_needs_no_attempt(self):
        agent = CorrectionAgent(rebuild=lambda p: passing_report())
        result = agent.correct(sample_params(), passing_report(), adjustable={"holes"})
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, [])

    def test_no_applicable_strategy_leaves_known_values_untouched(self):
        params = sample_params()
        report = fake_report({"bbox_axis_X": 70.0})
        agent = CorrectionAgent(rebuild=lambda p: report)
        result = agent.correct(params, report, adjustable=set())
        self.assertFalse(result.success)
        self.assertEqual(result.params, params)  # unchanged
        self.assertEqual(result.model_calls_made, 0)

    def test_unsolvable_known_contradiction(self):
        # Length is user-known; the only fix would weaken it: refuse.
        params = sample_params()
        report = fake_report({"bbox_axis_X": 70.0})
        seen = []

        def rebuild(candidate):
            seen.append(candidate)
            return report

        agent = CorrectionAgent(rebuild=rebuild)
        result = agent.correct(params, report, adjustable={"holes"})
        self.assertFalse(result.success)
        self.assertEqual(seen, [])  # never rebuilt: no valid strategy
        self.assertIn("will not be weakened", result.attempts[0].detail)

    def test_no_improvement_stops(self):
        # Reconcile produces a candidate (thickness 5 -> 4), but the rebuild
        # reports identical failures: stop after one fruitless attempt.
        params = sample_params()
        report = fake_report({"bbox_axis_Z": 4.0})
        calls = []

        def rebuild(candidate):
            calls.append(candidate)
            self.assertAlmostEqual(candidate.thickness, 4.0)
            return report  # identical failures: no progress possible

        agent = CorrectionAgent(rebuild=rebuild, max_attempts=2)
        result = agent.correct(params, report, adjustable={"thickness"})
        self.assertFalse(result.success)
        self.assertEqual(len(calls), 1)  # stopped after one fruitless attempt

    def test_attempt_limit_respected(self):
        # Custom check names the deterministic strategies ignore; the mock
        # model proposes a valid tweak each round while the fake rebuild
        # reports strictly shrinking failures, so only max_attempts stops it.
        params = sample_params()
        proposals = [{"thickness": 4.9}, {"thickness": 4.8}, {"thickness": 4.7}]

        def diagnose(**kwargs):
            return proposals.pop(0)

        diagnose.max_cost_usd = 1.0
        remaining = [{"check_b": 2.0, "check_c": 3.0}, {"check_c": 3.0}]

        def rebuild(candidate):
            return fake_report(remaining.pop(0))

        agent = CorrectionAgent(rebuild=rebuild, model_diagnose=diagnose,
                                max_attempts=2, max_correction_calls=5)
        start = fake_report({"check_a": 1.0, "check_b": 2.0, "check_c": 3.0})
        result = agent.correct(params, start, adjustable={"thickness"})
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.model_calls_made, 2)

    def test_unknown_adjustable_field_rejected(self):
        agent = CorrectionAgent(rebuild=lambda p: passing_report())
        with self.assertRaises(ValueError):
            agent.correct(sample_params(), fake_report({}), adjustable={"color"})


class TestModelCallDiscipline(TempDirTest):
    def test_no_model_call_for_deterministic_failure(self):
        calls = []

        def diagnose(**kwargs):
            calls.append(kwargs)
            return {}

        diagnose.max_cost_usd = 1.0
        params = sample_params()
        # Expected hole centre differs from the inferred one: deterministic
        # reconcile fixes it, so the hook must never fire.
        report = fake_report({"hole_0": {"x": 12, "y": 10, "diameter": 6}})
        agent = CorrectionAgent(rebuild=lambda p: passing_report(),
                                model_diagnose=diagnose)
        result = agent.correct(params, report, adjustable={"holes"})
        self.assertTrue(result.success)
        self.assertEqual(calls, [])
        self.assertEqual(result.model_calls_made, 0)

    def test_invalid_model_proposal_rejected_and_recorded(self):
        def diagnose(**kwargs):
            return {"thickness": -5}  # structurally invalid

        diagnose.max_cost_usd = 1.0
        params = sample_params()
        # A check name no deterministic strategy understands, so the invalid
        # model proposal is the only candidate in every round.
        report = fake_report({"custom_check": 1.0})
        agent = CorrectionAgent(rebuild=lambda p: passing_report(),
                                model_diagnose=diagnose, max_attempts=2)
        result = agent.correct(params, report, adjustable={"thickness"})
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 2)
        self.assertTrue(all("rejected" in a.detail for a in result.attempts))
        self.assertEqual(result.params, params)

    def test_model_proposal_cannot_touch_known_fields(self):
        def diagnose(**kwargs):
            return {"length": 70.0}  # length is NOT adjustable here

        diagnose.max_cost_usd = 1.0
        params = sample_params()
        report = fake_report({"bbox_axis_X": 70.0})
        agent = CorrectionAgent(rebuild=lambda p: passing_report(),
                                model_diagnose=diagnose, max_attempts=1)
        result = agent.correct(params, report, adjustable={"holes"})
        self.assertFalse(result.success)
        self.assertTrue(any("user-known" in a.detail for a in result.attempts))

    def test_correction_call_limit_enforced(self):
        calls = []

        def diagnose(**kwargs):
            calls.append(kwargs)
            return {}

        diagnose.max_cost_usd = 1.0
        params = sample_params()
        report = fake_report({"custom_check": 1.0})
        agent = CorrectionAgent(rebuild=lambda p: report,
                                model_diagnose=diagnose,
                                max_attempts=5, max_correction_calls=0)
        result = agent.correct(params, report, adjustable=set())
        self.assertEqual(calls, [])  # zero calls permitted: hook never fires
        self.assertEqual(result.model_calls_made, 0)
        self.assertTrue(any("limit" in a.detail for a in result.attempts))

    def test_budget_blocked_model_call_makes_no_call(self):
        calls = []

        def diagnose(**kwargs):
            calls.append(kwargs)
            return {}

        diagnose.max_cost_usd = 50.0  # exceeds the tiny test cap
        ledger = UsageLedger(self.path("ledger.jsonl"))
        guard = BudgetGuard(BudgetPolicy(per_experiment_cap_usd=1.0), ledger)
        params = sample_params()
        report = fake_report({"custom_check": 1.0})
        agent = CorrectionAgent(rebuild=lambda p: report,
                                model_diagnose=diagnose,
                                budget_guard=guard, max_attempts=1)
        result = agent.correct(params, report, adjustable=set())
        self.assertEqual(calls, [])  # the hook was never invoked
        self.assertTrue(any("budget guard" in a.detail for a in result.attempts))


class TestMissingHoleScenario(TempDirTest):
    """A required hole that cannot be established must never be invented.

    When the failing check names a hole index the candidate does not have,
    neither deterministic strategy can act (repair only moves existing holes;
    reconcile refuses out-of-range indices). The run must end unsuccessful
    with parameters byte-identical to the input: no invented hole, no
    silently dropped requirement.
    """

    def test_missing_hole_is_not_invented(self):
        params = validate_plate_params(
            length=60, width=40, thickness=5,
            holes=[(10, 10), (50, 10), (10, 30)], hole_diameter=6)
        report = fake_report({"hole_3": {"x": 50, "y": 30, "diameter": 6}})
        agent = CorrectionAgent(rebuild=lambda p: report)
        result = agent.correct(params, report, adjustable={"holes"})
        self.assertFalse(result.success)
        self.assertEqual(result.params, params)  # unchanged: nothing invented
        self.assertEqual(len(result.params.holes), 3)  # nothing added or removed
        self.assertEqual(result.model_calls_made, 0)
        self.assertLessEqual(len(result.attempts), 2)
        self.assertTrue(any(a.change == "no applicable strategy" for a in result.attempts))

    def test_missing_hole_never_weakens_user_values(self):
        params = validate_plate_params(
            length=60, width=40, thickness=5,
            holes=[(10, 10), (50, 10), (10, 30)], hole_diameter=6)
        report = fake_report({"hole_3": {"x": 50, "y": 30, "diameter": 6}})
        seen = []
        agent = CorrectionAgent(rebuild=lambda p: (seen.append(p), report)[1])
        result = agent.correct(params, report, adjustable=set())
        self.assertFalse(result.success)
        self.assertEqual(seen, [])  # no rebuild: no valid strategy exists
        self.assertEqual(result.params.to_dict(), params.to_dict())


class TestDiameterCorrectionScenario(TempDirTest):
    """A supported diameter correction targets the named hole parameter,
    preserves everything else, and honors user constraints."""

    @needs_openscad
    def test_wrong_diameter_corrected_to_named_hole(self):
        params = validate_plate_params(
            length=60, width=40, thickness=5,
            holes=[(10, 10, 8), (50, 10), (10, 30), (50, 30)],
            hole_diameter=6)
        agent = CorrectionAgent(rebuild=real_rebuild(self, CONSTRAINTS))
        initial = agent.rebuild(params)
        self.assertFalse(initial.passed)
        self.assertIn("hole_0", [c.name for c in initial.checks if not c.passed])
        result = agent.correct(params, initial, adjustable={"holes"})
        self.assertTrue(result.success, msg=str([a.detail for a in result.attempts]))
        # The named parameter was fixed: hole_0 is now exactly the constraint.
        self.assertEqual(result.params.holes[0].diameter, 6.0)
        self.assertEqual((result.params.holes[0].x, result.params.holes[0].y),
                         (10.0, 10.0))
        # Everything unrelated is preserved.
        self.assertEqual((result.params.length, result.params.width,
                          result.params.thickness), (60.0, 40.0, 5.0))
        self.assertEqual(
            [(h.x, h.y, h.diameter) for h in result.params.holes[1:]],
            [(50.0, 10.0, 6.0), (10.0, 30.0, 6.0), (50.0, 30.0, 6.0)])
        self.assertEqual(result.model_calls_made, 0)
        self.assertLessEqual(len(result.attempts), 2)
        # The corrected artifact passes the real validation path.
        self.assertTrue(result.report.passed)

    def test_diameter_user_constraint_is_not_overridden(self):
        params = validate_plate_params(
            length=60, width=40, thickness=5,
            holes=[(10, 10, 8), (50, 10), (10, 30), (50, 30)],
            hole_diameter=6)
        report = fake_report({"hole_0": {"x": 10, "y": 10, "diameter": 6}})
        agent = CorrectionAgent(rebuild=lambda p: report)
        result = agent.correct(params, report, adjustable=set())
        self.assertFalse(result.success)
        self.assertEqual(result.params.holes[0].diameter, 8.0)  # untouched


class TestHookFailureScenario(TempDirTest):
    """Hook failures are reported honestly: recorded, never fabricated."""

    def test_raising_hook_is_recorded_not_fabricated(self):
        def diagnose(**kwargs):
            raise RuntimeError("model unavailable")

        params = sample_params()
        report = fake_report({"custom_check": 1.0})
        agent = CorrectionAgent(rebuild=lambda p: passing_report(),
                                model_diagnose=diagnose, max_attempts=2)
        result = agent.correct(params, report, adjustable=set())
        self.assertFalse(result.success)
        self.assertEqual(result.model_calls_made, 0)  # the call never completed
        self.assertEqual(result.params, params)  # unchanged
        self.assertTrue(any("diagnose hook failed: model unavailable" in a.detail
                            for a in result.attempts))

    def test_garbage_hook_result_is_rejected_not_applied(self):
        def diagnose(**kwargs):
            return ["not", "a", "mapping"]  # invalid hook result

        params = sample_params()
        report = fake_report({"custom_check": 1.0})
        agent = CorrectionAgent(rebuild=lambda p: passing_report(),
                                model_diagnose=diagnose, max_attempts=2)
        result = agent.correct(params, report, adjustable=set())
        self.assertFalse(result.success)
        self.assertEqual(result.params, params)  # unchanged: nothing applied
        self.assertTrue(all("rejected" in a.detail for a in result.attempts))
        self.assertLessEqual(len(result.attempts), 2)  # still bounded


class TestBoundedCorrectionInvariants(unittest.TestCase):
    def test_default_attempt_limit_is_two(self):
        agent = CorrectionAgent(rebuild=lambda p: passing_report())
        self.assertEqual(agent.max_attempts, 2)  # original prompt maximum

    def test_only_named_fields_are_adjustable(self):
        from reforge.correction_agent import ADJUSTABLE_FIELDS

        self.assertEqual(ADJUSTABLE_FIELDS, {"length", "width", "thickness", "holes"})


if __name__ == "__main__":
    unittest.main()
