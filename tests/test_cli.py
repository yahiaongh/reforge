"""CLI tests: generate/validate/pipeline/budget wiring (no paid calls)."""

import json
import os
import unittest

from tests import TempDirTest, needs_openscad
from reforge.cli import main


class TestGenerateAndBudget(TempDirTest):
    def test_generate_scad_only(self):
        code = main(["generate", "--length", "60", "--width", "40",
                     "--thickness", "5", "--hole-diameter", "6",
                     "--holes", "10,10;50,10;10,30;50,30",
                     "--scad", self.path("plate.scad")])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(self.path("plate.scad")))

    def test_generate_rejects_bad_params(self):
        code = main(["generate", "--length", "-60", "--width", "40",
                     "--thickness", "5", "--scad", self.path("plate.scad")])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(self.path("plate.scad")))

    def test_budget_reports_empty_ledger(self):
        code = main(["budget", "--ledger", self.path("usage.jsonl")])
        self.assertEqual(code, 0)


class TestPipelineCli(TempDirTest):
    @needs_openscad
    def test_pipeline_self_check_passes(self):
        code = main(["pipeline", "--length", "60", "--width", "40",
                     "--thickness", "5", "--hole-diameter", "6",
                     "--holes", "10,10;50,10;10,30;50,30",
                     "--outdir", self.path("out")])
        self.assertEqual(code, 0)
        for name in ("plate.scad", "plate.stl", "validation_report.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.path("out"), name)))

    @needs_openscad
    def test_pipeline_correction_demo(self):
        # Candidate has an inferred hole away from the known constraint;
        # --correct must reconcile it and finish green.
        constraints = {
            "length": 60, "width": 40, "thickness": 5,
            "holes": [{"x": 10, "y": 10, "diameter": 6}],
        }
        constraints_path = self.path("constraints.json")
        with open(constraints_path, "w", encoding="utf-8") as handle:
            json.dump(constraints, handle)
        code = main(["pipeline", "--length", "60", "--width", "40",
                     "--thickness", "5",
                     "--holes", "14,10,6",
                     "--constraints-json", constraints_path,
                     "--correct", "--adjustable", "holes",
                     "--outdir", self.path("out")])
        self.assertEqual(code, 0)
        correction_log = os.path.join(self.path("out"), "correction_log.json")
        self.assertTrue(os.path.isfile(correction_log))
        with open(correction_log, encoding="utf-8") as handle:
            logged = json.load(handle)
        self.assertTrue(logged["success"])
        self.assertEqual(logged["model_calls_made"], 0)
        self.assertEqual(logged["final_params"]["holes"],
                         [{"x": 10.0, "y": 10.0, "diameter": 6.0}])

    @needs_openscad
    def test_pipeline_fails_without_correct(self):
        constraints = {
            "length": 60, "width": 40, "thickness": 5,
            "holes": [{"x": 10, "y": 10, "diameter": 6}],
        }
        constraints_path = self.path("constraints.json")
        with open(constraints_path, "w", encoding="utf-8") as handle:
            json.dump(constraints, handle)
        code = main(["pipeline", "--length", "60", "--width", "40",
                     "--thickness", "5",
                     "--holes", "14,10,6",
                     "--constraints-json", constraints_path,
                     "--outdir", self.path("out")])
        self.assertEqual(code, 1)  # validation failed, no correction requested


if __name__ == "__main__":
    unittest.main()
