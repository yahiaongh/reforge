"""Parameter validation tests: valid sets plus every rejection class."""

import math
import unittest

from tests import sample_params
from reforge.parameters import (
    ParameterError,
    plate_params_from_dict,
    validate_plate_params,
)


class TestValidParameters(unittest.TestCase):
    def test_sample_plate_is_valid(self):
        params = sample_params()
        self.assertEqual(params.length, 60.0)
        self.assertEqual(params.width, 40.0)
        self.assertEqual(params.thickness, 5.0)
        self.assertEqual(len(params.holes), 4)

    def test_plate_without_holes_is_valid(self):
        params = validate_plate_params(length=20, width=10, thickness=2)
        self.assertEqual(params.holes, ())

    def test_explicit_per_hole_diameters(self):
        params = validate_plate_params(
            length=60, width=40, thickness=5,
            holes=[(10, 10, 6), (50, 30, 8)],
        )
        self.assertEqual(params.holes[1].diameter, 8.0)

    def test_hole_tangent_to_edge_is_allowed(self):
        params = validate_plate_params(
            length=20, width=20, thickness=3,
            holes=[(5, 5, 10)],  # circle exactly spans 0..10 in x
        )
        self.assertEqual(len(params.holes), 1)


class TestDimensionRejections(unittest.TestCase):
    def test_zero_dimension_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=0, width=10, thickness=2)

    def test_negative_dimensions_rejected(self):
        for kwargs in (
            dict(length=-5, width=10, thickness=2),
            dict(length=5, width=-10, thickness=2),
            dict(length=5, width=10, thickness=-2),
        ):
            with self.assertRaises(ParameterError, msg=kwargs):
                validate_plate_params(**kwargs)

    def test_non_finite_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ParameterError, msg=repr(bad)):
                validate_plate_params(length=bad, width=10, thickness=2)

    def test_bool_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=True, width=10, thickness=2)

    def test_string_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length="60", width=10, thickness=2)

    def test_none_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=None, width=10, thickness=2)

    def test_oversize_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=1e9, width=10, thickness=2)


class TestHoleRejections(unittest.TestCase):
    def test_zero_hole_diameter_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=20, width=20, thickness=2,
                                  holes=[(10, 10)], hole_diameter=0)

    def test_negative_hole_diameter_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=20, width=20, thickness=2,
                                  holes=[(10, 10)], hole_diameter=-3)

    def test_hole_outside_plate_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=20, width=20, thickness=2,
                                  holes=[(19, 10)], hole_diameter=6)

    def test_hole_larger_than_plate_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=20, width=20, thickness=2,
                                  holes=[(10, 10)], hole_diameter=30)

    def test_overlapping_holes_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=60, width=40, thickness=5,
                                  holes=[(10, 10), (12, 10)], hole_diameter=6)

    def test_malformed_hole_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=20, width=20, thickness=2,
                                  holes=["not-a-hole"], hole_diameter=4)

    def test_missing_hole_diameter_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=20, width=20, thickness=2,
                                  holes=[(10, 10)])

    def test_unknown_hole_key_rejected(self):
        with self.assertRaises(ParameterError):
            validate_plate_params(length=20, width=20, thickness=2,
                                  holes=[{"x": 10, "y": 10, "diameter": 4, "depth": 2}])


class TestFromDict(unittest.TestCase):
    def test_round_trip(self):
        raw = {
            "length": 60, "width": 40, "thickness": 5,
            "holes": [{"x": 10, "y": 10, "diameter": 6}],
        }
        params = plate_params_from_dict(raw)
        self.assertEqual(params.to_dict()["holes"], raw["holes"])

    def test_missing_key_rejected(self):
        with self.assertRaises(ParameterError):
            plate_params_from_dict({"length": 60, "width": 40})

    def test_unknown_key_rejected(self):
        with self.assertRaises(ParameterError):
            plate_params_from_dict(
                {"length": 60, "width": 40, "thickness": 5, "color": "red"}
            )

    def test_none_dimension_rejected(self):
        with self.assertRaises(ParameterError):
            plate_params_from_dict({"length": None, "width": 40, "thickness": 5})

    def test_non_mapping_rejected(self):
        with self.assertRaises(ParameterError):
            plate_params_from_dict([("length", 60)])


if __name__ == "__main__":
    unittest.main()
