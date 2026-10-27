"""Validator tests: real meshes plus failure and tolerance-boundary cases."""

import os
import unittest

from tests import TempDirTest, needs_openscad, sample_params
from reforge.cad_generator import write_scad
from reforge.parameters import validate_plate_params
from reforge.renderer import render_scad_to_stl
from reforge.validator import read_stl_triangles, validate_stl


def render_sample(case: TempDirTest, **overrides):
    """Render the sample plate (or overridden params) into the temp dir."""
    from tests import SAMPLE_KWARGS

    kwargs = dict(SAMPLE_KWARGS)
    kwargs.update(overrides)
    params = validate_plate_params(**kwargs)
    scad = case.path("plate.scad")
    stl = case.path("plate.stl")
    write_scad(params, scad)
    render_scad_to_stl(scad, stl)
    return stl, params


class TestValidatorFailures(TempDirTest):
    def test_missing_file_fails(self):
        report = validate_stl(self.path("nope.stl"), sample_params())
        self.assertFalse(report.passed)
        self.assertEqual(report.checks[0].name, "file_present")

    def test_empty_file_fails(self):
        path = self.path("empty.stl")
        open(path, "wb").close()
        report = validate_stl(path, sample_params())
        self.assertFalse(report.passed)

    def test_garbage_file_fails_parse(self):
        path = self.path("garbage.stl")
        with open(path, "wb") as handle:
            handle.write(b"\x00" * 200)
        report = validate_stl(path, sample_params())
        self.assertFalse(report.passed)
        names = [c.name for c in report.checks]
        self.assertIn("stl_parseable", names)

    def test_truncated_ascii_stl_rejected(self):
        path = self.path("trunc.stl")
        with open(path, "w", encoding="ascii") as handle:
            handle.write("solid trunc\n  facet normal 0 0 1\n    outer loop\n")
        with self.assertRaises(ValueError):
            read_stl_triangles(path)


class TestValidatorOnRealMesh(TempDirTest):
    @needs_openscad
    def test_valid_mesh_passes(self):
        stl, params = render_sample(self)
        report = validate_stl(stl, params)
        self.assertTrue(report.passed, msg=report.human_summary())
        self.assertGreater(report.facet_count, 0)

    @needs_openscad
    def test_wrong_expected_dimensions_fail_bbox(self):
        stl, _ = render_sample(self)
        wrong = validate_plate_params(
            length=70, width=40, thickness=5,
            holes=[(10, 10), (50, 10), (10, 30), (50, 30)], hole_diameter=6,
        )
        report = validate_stl(stl, wrong)
        self.assertFalse(report.passed)
        by_name = {c.name: c for c in report.checks}
        self.assertFalse(by_name["bbox_axis_X"].passed)
        self.assertTrue(by_name["bbox_axis_Y"].passed)

    @needs_openscad
    def test_wrong_hole_diameter_fails(self):
        stl, _ = render_sample(self)
        wrong = validate_plate_params(
            length=60, width=40, thickness=5,
            holes=[(10, 10), (50, 10), (10, 30), (50, 30)], hole_diameter=10,
        )
        report = validate_stl(stl, wrong)
        self.assertFalse(report.passed)
        self.assertFalse(report.checks[-1].passed)

    @needs_openscad
    def test_tolerance_boundary(self):
        # The outer faces are exact, so the bbox passes even at 1e-6; the hole
        # diameter carries faceting + ray-step error (~0.05 mm here), so an
        # impossibly tight diameter tolerance must fail while a generous one
        # passes. This proves the tolerances are actually applied.
        stl, params = render_sample(self)
        exact_bbox = validate_stl(stl, params, bbox_tol=1e-6, hole_diam_tol=5.0)
        self.assertTrue(exact_bbox.passed, msg=exact_bbox.human_summary())
        strict = validate_stl(stl, params, hole_diam_tol=1e-6)
        self.assertFalse(strict.passed)
        loose = validate_stl(stl, params, hole_diam_tol=5.0)
        self.assertTrue(loose.passed, msg=loose.human_summary())

    @needs_openscad
    def test_report_is_machine_readable(self):
        stl, params = render_sample(self)
        report = validate_stl(stl, params)
        payload = report.to_dict()
        self.assertIn("passed", payload)
        self.assertIn("checks", payload)
        self.assertTrue(all("name" in c and "passed" in c for c in payload["checks"]))
        hole_checks = [c for c in payload["checks"] if c["name"].startswith("hole_")]
        self.assertEqual(len(hole_checks), 4)
        for check in hole_checks:
            self.assertIn("measured", check)


if __name__ == "__main__":
    unittest.main()
