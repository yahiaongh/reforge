"""CAD generator tests: substitution, placement, syntax shape, injection safety."""

import re
import unittest

from tests import sample_params
from reforge.cad_generator import format_number, generate_scad, write_scad
from reforge.parameters import ParameterError, validate_plate_params
from tests import TempDirTest


class TestGeneration(unittest.TestCase):
    def test_plate_dimensions_substituted(self):
        source = generate_scad(sample_params())
        self.assertIn("cube([60.000000, 40.000000, 5.000000]", source)

    def test_all_holes_placed(self):
        source = generate_scad(sample_params())
        for x, y in ((10, 10), (50, 10), (10, 30), (50, 30)):
            self.assertIn(f"translate([{x:.6f}, {y:.6f},", source)
        self.assertEqual(source.count("cylinder("), 4)

    def test_hole_diameter_substituted(self):
        source = generate_scad(sample_params())
        self.assertIn("d = 6.000000", source)

    def test_cutter_spans_full_thickness(self):
        # Cutter height must exceed the plate thickness (overhang both sides).
        source = generate_scad(sample_params())
        match = re.search(r"cylinder\(h = ([\d.]+),", source)
        self.assertIsNotNone(match)
        self.assertGreater(float(match.group(1)), 5.0)

    def test_deterministic_output(self):
        self.assertEqual(generate_scad(sample_params()), generate_scad(sample_params()))

    def test_no_holes_means_plain_block(self):
        params = validate_plate_params(length=20, width=10, thickness=2)
        source = generate_scad(params)
        self.assertIn("cube(", source)
        self.assertNotIn("cylinder(", source)

    def test_write_scad_round_trip(self):
        case = TempDirTest()
        case.setUp()
        self.addCleanup(case.tmp.cleanup)
        path = case.path("plate.scad")
        source = write_scad(sample_params(), path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), source)


class TestInjectionSafety(unittest.TestCase):
    def test_numeric_formatting_has_no_code_characters(self):
        for value in (60.0, 0.1, 1e-3, 9999.999):
            token = format_number(value)
            self.assertRegex(token, r"^-?\d+\.\d+$", msg=token)

    def test_malicious_strings_never_reach_template(self):
        # Validation rejects non-numeric input, so hostile strings cannot be
        # interpolated: the generator only ever receives validated floats.
        with self.assertRaises(ParameterError):
            validate_plate_params(length="60]); evil(); //", width=10, thickness=2)
        source = generate_scad(sample_params())
        for suspicious in ("evil", "import", "exec", "system", "echo"):
            self.assertNotIn(suspicious, source)

    def test_only_expected_scad_primitives(self):
        source = generate_scad(sample_params())
        identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source))
        allowed = {
            "Generated", "by", "Reforge", "from", "validated", "parameters",
            "Editable", "plate", "hole", "holes", "difference", "cube",
            "center", "false", "translate", "cylinder", "mm", "x",
            # Trusted-template parameter names and the "hole(s)" plural.
            "d", "h", "fn", "s",
        }
        self.assertTrue(identifiers <= allowed, f"unexpected tokens: {identifiers - allowed}")


if __name__ == "__main__":
    unittest.main()
