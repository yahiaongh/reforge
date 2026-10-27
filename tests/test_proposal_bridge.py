"""Proposal-to-parameters bridge tests. Fully offline: no network, no models."""

import unittest

from reforge.geometry import ProposalError, validate_proposal
from reforge.proposal_bridge import BridgeError, convert_proposal

try:
    from test_geometry import measurement, valid_proposal_dict
except ImportError:  # run as tests.test_proposal_bridge instead of discover
    from tests.test_geometry import measurement, valid_proposal_dict


def convertible_proposal_dict():
    """A fully user-confirmed, scale-grounded proposal ready for conversion."""
    raw = valid_proposal_dict()

    def confirmed(value, uncertainty_mm=0.0):
        return {"value": value, "confidence": 1.0, "uncertainty_mm": uncertainty_mm,
                "source": "user_confirmed", "requires_confirmation": False}

    raw["length"] = confirmed(58.0, uncertainty_mm=0.05)
    raw["width"] = confirmed(38.0)
    raw["thickness"] = confirmed(5.0)
    for hole in raw["holes"]:
        hole["x"] = confirmed(hole["x"]["value"])
        hole["y"] = confirmed(hole["y"]["value"])
        hole["diameter"] = confirmed(hole["diameter"]["value"])
        hole["depth"] = "through"
    raw["scale"] = {"method": "user_measured_reference",
                    "detail": "caliper measurement by operator"}
    raw["requires_confirmation"] = []
    return raw


def assert_bridge_rejected(test, raw, fragment):
    with test.assertRaises(BridgeError) as ctx:
        convert_proposal(raw)
    test.assertIn(fragment.lower(), str(ctx.exception).lower(),
                  msg=str(ctx.exception))
    return ctx.exception


class TestSuccessfulConversion(unittest.TestCase):
    def test_valid_proposal_converts_with_exact_values(self):
        result = convert_proposal(convertible_proposal_dict())
        params = result.params
        self.assertEqual((params.length, params.width, params.thickness),
                         (58.0, 38.0, 5.0))
        self.assertEqual(
            [(h.x, h.y, h.diameter) for h in params.holes],
            [(10.0, 10.0, 6.0), (48.0, 10.0, 6.0),
             (10.0, 28.0, 6.0), (48.0, 28.0, 6.0)])

    def test_provenance_preserved_without_upgrade(self):
        result = convert_proposal(convertible_proposal_dict())
        length = result.provenance["length"]
        self.assertEqual(length.source, "user_confirmed")
        self.assertEqual(length.confidence, 1.0)
        # Instrument tolerance passes through unchanged, never zeroed or grown.
        self.assertEqual(length.uncertainty_mm, 0.05)
        self.assertFalse(length.requires_confirmation)
        self.assertEqual(result.scale_method, "user_measured_reference")
        self.assertIn("note", result.evidence_summary)
        self.assertIn("never converted to millimetres",
                      result.evidence_summary["note"])

    def test_accepts_validated_proposal_object(self):
        proposal = validate_proposal(convertible_proposal_dict())
        result = convert_proposal(proposal)
        self.assertEqual(result.params.length, 58.0)

    def test_ungrounded_scale_ok_when_all_user_confirmed(self):
        raw = convertible_proposal_dict()
        raw["scale"] = {"method": "unknown", "detail": "no image scale used"}
        raw["requires_confirmation"] = ["scale"]
        # Scale listing without grounding: confirmation rule demands the flag
        # only when ungrounded, which holds here; conversion needs no image scale.
        result = convert_proposal(raw)
        self.assertEqual(result.params.width, 38.0)

    def test_invalid_mapping_raises_proposal_error(self):
        raw = convertible_proposal_dict()
        del raw["length"]
        with self.assertRaises(ProposalError):
            convert_proposal(raw)

    def test_parameter_wrap_turns_into_bridge_error(self):
        import reforge.proposal_bridge as bridge
        from reforge.parameters import ParameterError

        real = bridge.validate_plate_params
        try:
            def failing(**kwargs):
                raise ParameterError("simulated downstream rejection")

            bridge.validate_plate_params = failing
            with self.assertRaises(BridgeError) as ctx:
                convert_proposal(convertible_proposal_dict())
            self.assertIn("parameter constraints", str(ctx.exception))
        finally:
            bridge.validate_plate_params = real


class TestBridgeRejections(unittest.TestCase):
    def test_missing_thickness_rejected(self):
        raw = valid_proposal_dict()  # thickness null, confirmations pending
        exc = assert_bridge_rejected(self, raw, "thickness")
        self.assertTrue(any("confirmation" in r for r in exc.reasons))

    def test_unconfirmed_inferred_measurements_rejected(self):
        raw = convertible_proposal_dict()
        raw["holes"][0]["x"] = measurement(
            10.0, confidence=0.9, uncertainty_mm=0.5,
            source="model_inferred", requires_confirmation=True)
        raw["requires_confirmation"] = ["holes[0].x"]
        exc = assert_bridge_rejected(self, raw, "holes[0].x")
        self.assertTrue(any("confirmation" in r for r in exc.reasons))

    def test_ungrounded_scale_with_estimates_rejected(self):
        raw = convertible_proposal_dict()
        raw["scale"] = {"method": "assumed", "detail": "assumed 10 px/mm"}
        raw["requires_confirmation"] = ["scale"]
        # Measurements are user_confirmed here, so scale is moot: must still
        # reject when estimates are involved; rebuild with inferred values.
        raw["length"] = measurement(58.0, confidence=0.9, uncertainty_mm=0.5,
                                    source="scale_estimate",
                                    requires_confirmation=False)
        assert_bridge_rejected(self, raw, "scale")

    def test_null_hole_depth_rejected(self):
        raw = convertible_proposal_dict()
        raw["holes"][2]["depth"] = None
        raw["requires_confirmation"] = ["holes[2]"]
        assert_bridge_rejected(self, raw, "holes[2]")

    def test_unsupported_classification_rejected(self):
        raw = convertible_proposal_dict()
        raw["classification"] = "washer"
        with self.assertRaises(ProposalError):
            convert_proposal(raw)

    def test_no_invented_defaults(self):
        # Every rejection above leaves no params behind: the exception carries
        # reasons, never a half-filled parameter set.
        raw = valid_proposal_dict()
        try:
            convert_proposal(raw)
            self.fail("expected BridgeError")
        except BridgeError as exc:
            self.assertGreaterEqual(len(exc.reasons), 2)
            self.assertFalse(hasattr(exc, "params"))


class TestDownstreamCompatibility(unittest.TestCase):
    def test_generated_scad_matches_converted_values(self):
        from reforge.cad_generator import generate_scad

        result = convert_proposal(convertible_proposal_dict())
        source = generate_scad(result.params)
        self.assertIn("cube([58.000000, 38.000000, 5.000000]", source)
        self.assertEqual(source.count("cylinder("), 4)

    def test_render_and_validate_converted_params(self):
        import shutil

        from tests import TempDirTest

        if shutil.which("openscad") is None:
            self.skipTest("OpenSCAD executable not found")
        from reforge.cad_generator import write_scad
        from reforge.renderer import render_scad_to_stl
        from reforge.validator import validate_stl

        case = TempDirTest()
        case.setUp()
        self.addCleanup(case.tmp.cleanup)
        result = convert_proposal(convertible_proposal_dict())
        scad, stl = case.path("plate.scad"), case.path("plate.stl")
        write_scad(result.params, scad)
        render_scad_to_stl(scad, stl)
        report = validate_stl(stl, result.params)
        self.assertTrue(report.passed, msg=report.human_summary())


if __name__ == "__main__":
    unittest.main()
