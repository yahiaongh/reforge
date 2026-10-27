"""Geometry proposal schema tests. Fully offline: no network, no models."""

import copy
import unittest

from reforge.geometry import (
    GeometryProposal,
    ProposalError,
    parse_proposal_json,
    validate_proposal,
)


def measurement(value, confidence=0.9, uncertainty_mm=0.5,
                source="model_inferred", requires_confirmation=True):
    return {
        "value": value,
        "confidence": confidence,
        "uncertainty_mm": uncertainty_mm,
        "source": source,
        "requires_confirmation": requires_confirmation,
    }


def valid_proposal_dict():
    proposal = {
        "classification": "rectangular_plate",
        "units": "mm",
        "conventions": {
            "origin": "plate_corner",
            "axes": {"x": "length", "y": "width", "z": "thickness"},
        },
        "length": measurement(58.0),
        "width": measurement(38.0),
        "thickness": {
            "value": None, "confidence": 0.0, "uncertainty_mm": None,
            "source": "model_inferred", "requires_confirmation": True,
        },
        "holes": [
            {"x": measurement(10.0), "y": measurement(10.0),
             "diameter": measurement(6.0), "depth": "through"},
            {"x": measurement(48.0), "y": measurement(10.0),
             "diameter": measurement(6.0), "depth": "through"},
            {"x": measurement(10.0), "y": measurement(28.0),
             "diameter": measurement(6.0), "depth": "through"},
            {"x": measurement(48.0), "y": measurement(28.0),
             "diameter": measurement(6.0), "depth": "through"},
        ],
        "scale": {"method": "ruler_in_image",
                  "detail": "steel ruler visible along the bottom edge"},
        "assumptions": ["plate face approximately parallel to the image plane"],
        "unknowns": ["thickness is not visible in a top-down photograph"],
        "requires_confirmation": [],
        "perspective_warning": "slight off-axis angle; edges may be biased by ~1-2%",
        "evidence": {
            "image_asset_hash": "ab" * 32,
            "mask_bounds": [415.0, 201.0, 993.0, 521.0],
            "box_confidence": 0.95,
        },
    }
    flagged = ["length", "width", "thickness", "scale"]
    for i in range(4):
        flagged += [f"holes[{i}].x", f"holes[{i}].y", f"holes[{i}].diameter"]
    proposal["requires_confirmation"] = flagged
    return proposal


def assert_rejected(test, raw, field_fragment, phase=None):
    with test.assertRaises(ProposalError) as ctx:
        validate_proposal(raw)
    message = str(ctx.exception).lower()
    test.assertIn(field_fragment.lower(), message, msg=str(ctx.exception))
    if phase is not None:
        phases = {issue.phase for issue in ctx.exception.issues}
        test.assertIn(phase, phases)
    return ctx.exception


class TestValidProposal(unittest.TestCase):
    def test_complete_proposal_accepted(self):
        proposal = validate_proposal(valid_proposal_dict())
        self.assertIsInstance(proposal, GeometryProposal)
        self.assertEqual(proposal.classification, "rectangular_plate")
        self.assertEqual(proposal.units, "mm")
        self.assertEqual(proposal.origin, "plate_corner")
        self.assertEqual(len(proposal.holes), 4)
        self.assertIsNone(proposal.thickness.value)  # unknown stays unknown
        self.assertEqual(proposal.scale_method, "ruler_in_image")

    def test_ordinary_prose_with_geometry_words_accepted(self):
        # Bare words without code-shaped calls must not trip the screen.
        raw = valid_proposal_dict()
        raw["assumptions"] = ["edge finish resembles a brushed cube at small scale"]
        raw["unknowns"] = ["whether the fourth hole needs a translate of the grid"]
        validate_proposal(raw)  # must not raise

    def test_user_confirmed_certain_proposal_accepted(self):
        raw = valid_proposal_dict()

        def confirmed(value):
            return {"value": value, "confidence": 1.0, "uncertainty_mm": 0.0,
                    "source": "user_confirmed", "requires_confirmation": False}

        raw["length"] = confirmed(58.0)
        raw["width"] = confirmed(38.0)
        raw["thickness"] = confirmed(5.0)
        for hole in raw["holes"]:
            hole["x"] = confirmed(hole["x"]["value"])
            hole["y"] = confirmed(hole["y"]["value"])
            hole["diameter"] = confirmed(hole["diameter"]["value"])
        raw["scale"] = {"method": "user_measured_reference",
                        "detail": "caliper measurement by operator"}
        raw["requires_confirmation"] = []
        proposal = validate_proposal(raw)
        self.assertEqual(proposal.thickness.value, 5.0)


class TestStructuralRejections(unittest.TestCase):
    def test_non_mapping_rejected(self):
        with self.assertRaises(ProposalError):
            validate_proposal([("classification", "rectangular_plate")])

    def test_missing_required_dimensions(self):
        raw = valid_proposal_dict()
        del raw["width"]
        assert_rejected(self, raw, "width", phase="structural")

    def test_null_length_rejected(self):
        raw = valid_proposal_dict()
        raw["length"]["value"] = None
        assert_rejected(self, raw, "length", phase="structural")

    def test_unsupported_units(self):
        raw = valid_proposal_dict()
        raw["units"] = "cm"
        assert_rejected(self, raw, "units", phase="structural")

    def test_missing_units(self):
        raw = valid_proposal_dict()
        del raw["units"]
        assert_rejected(self, raw, "units", phase="structural")

    def test_invalid_origin(self):
        raw = valid_proposal_dict()
        raw["conventions"]["origin"] = "image_center"
        assert_rejected(self, raw, "conventions.origin", phase="structural")

    def test_invalid_axes(self):
        raw = valid_proposal_dict()
        raw["conventions"]["axes"] = {"x": "width", "y": "length", "z": "thickness"}
        assert_rejected(self, raw, "conventions.axes", phase="structural")

    def test_missing_hole_geometry(self):
        raw = valid_proposal_dict()
        del raw["holes"]
        assert_rejected(self, raw, "holes", phase="structural")

    def test_empty_holes_rejected(self):
        raw = valid_proposal_dict()
        raw["holes"] = []
        raw["requires_confirmation"] = ["length", "width", "thickness", "scale"]
        assert_rejected(self, raw, "holes", phase="structural")

    def test_unknown_classification(self):
        raw = valid_proposal_dict()
        raw["classification"] = "engine_block"
        assert_rejected(self, raw, "classification", phase="structural")

    def test_recognized_but_ungeneratable_classification(self):
        raw = valid_proposal_dict()
        raw["classification"] = "washer"
        exc = assert_rejected(self, raw, "classification", phase="structural")
        self.assertIn("cannot represent", str(exc))

    def test_unknown_top_level_key(self):
        raw = valid_proposal_dict()
        raw["material"] = "steel"
        assert_rejected(self, raw, "unknown top-level keys", phase="structural")

    def test_unsupported_depth(self):
        raw = valid_proposal_dict()
        raw["holes"][0]["depth"] = "blind"
        assert_rejected(self, raw, "depth", phase="structural")

    def test_oversized_hole_list_rejected(self):
        raw = valid_proposal_dict()
        raw["holes"] = [copy.deepcopy(raw["holes"][0]) for _ in range(26)]
        assert_rejected(self, raw, "holes", phase="structural")

    def test_executable_content_rejected(self):
        raw = valid_proposal_dict()
        raw["assumptions"] = ["holes look symmetric __import__('os').system('x')"]
        assert_rejected(self, raw, "assumptions", phase="structural")

    def test_cad_primitive_in_prose_rejected(self):
        for marker, field_name in (("cube([58, 38, 5])", "assumptions"),
                                   ("difference() the plate", "unknowns")):
            raw = valid_proposal_dict()
            raw[field_name] = [f"supposedly just text with {marker} inside"]
            assert_rejected(self, raw, field_name, phase="structural")

    def test_malformed_values(self):
        raw = valid_proposal_dict()
        raw["length"]["value"] = "fifty-eight"
        assert_rejected(self, raw, "length", phase="structural")
        raw = valid_proposal_dict()
        raw["length"]["value"] = True
        assert_rejected(self, raw, "length", phase="structural")
        raw = valid_proposal_dict()
        raw["length"]["value"] = float("nan")
        assert_rejected(self, raw, "length", phase="structural")

    def test_invalid_source(self):
        raw = valid_proposal_dict()
        raw["length"]["source"] = "vibes"
        assert_rejected(self, raw, "length", phase="structural")

    def test_bad_evidence(self):
        raw = valid_proposal_dict()
        raw["evidence"]["image_asset_hash"] = "not-a-hash"
        assert_rejected(self, raw, "evidence.image_asset_hash", phase="structural")


class TestPlausibilityRejections(unittest.TestCase):
    def test_confidence_out_of_range(self):
        raw = valid_proposal_dict()
        raw["length"]["confidence"] = 1.5
        assert_rejected(self, raw, "length", phase="plausibility")
        raw = valid_proposal_dict()
        raw["width"]["confidence"] = -0.1
        assert_rejected(self, raw, "width", phase="plausibility")

    def test_negative_uncertainty(self):
        raw = valid_proposal_dict()
        raw["length"]["uncertainty_mm"] = -0.5
        assert_rejected(self, raw, "length", phase="plausibility")

    def test_uncertain_claim_without_quantified_uncertainty(self):
        raw = valid_proposal_dict()
        raw["length"]["uncertainty_mm"] = None
        assert_rejected(self, raw, "uncertainty_mm", phase="plausibility")

    def test_certainty_with_tolerance_accepted(self):
        # Confidence 1.0 with a positive uncertainty is a legitimate
        # certain-within-tolerance claim, not a contradiction.
        raw = valid_proposal_dict()
        raw["length"]["confidence"] = 1.0
        raw["length"]["uncertainty_mm"] = 0.5
        raw["length"]["source"] = "user_confirmed"
        raw["length"]["requires_confirmation"] = False
        raw["requires_confirmation"] = [c for c in raw["requires_confirmation"]
                                        if c != "length"]
        proposal = validate_proposal(raw)
        self.assertEqual(proposal.length.confidence, 1.0)
        self.assertEqual(proposal.length.uncertainty_mm, 0.5)

    def test_user_confirmed_caliper_reading_with_precision(self):
        # Regression: a user-confirmed measurement carrying its instrument
        # tolerance must validate (previously rejected by the 1.0 coupling).
        raw = valid_proposal_dict()
        raw["thickness"] = {"value": 5.0, "confidence": 1.0,
                            "uncertainty_mm": 0.05, "source": "user_confirmed",
                            "requires_confirmation": False}
        raw["requires_confirmation"] = [c for c in raw["requires_confirmation"]
                                        if c != "thickness"]
        proposal = validate_proposal(raw)
        self.assertEqual(proposal.thickness.value, 5.0)
        self.assertEqual(proposal.thickness.uncertainty_mm, 0.05)

    def test_zero_confidence_with_stated_value(self):
        raw = valid_proposal_dict()
        raw["length"]["confidence"] = 0.0
        assert_rejected(self, raw, "length", phase="plausibility")

    def test_null_thickness_with_nonzero_confidence(self):
        raw = valid_proposal_dict()
        raw["thickness"]["confidence"] = 0.7
        assert_rejected(self, raw, "thickness", phase="plausibility")

    def test_negative_dimension(self):
        raw = valid_proposal_dict()
        raw["length"]["value"] = -58.0
        assert_rejected(self, raw, "length", phase="plausibility")

    def test_hole_outside_plate(self):
        raw = valid_proposal_dict()
        raw["holes"][0]["x"]["value"] = 57.0
        assert_rejected(self, raw, "holes[0]", phase="plausibility")

    def test_overlapping_holes(self):
        raw = valid_proposal_dict()
        raw["holes"][1]["x"]["value"] = 12.0
        assert_rejected(self, raw, "holes[", phase="plausibility")

    def test_oversize_hole_diameter(self):
        raw = valid_proposal_dict()
        raw["holes"][0]["diameter"]["value"] = 60.0
        assert_rejected(self, raw, "holes[0].diameter", phase="plausibility")

    def test_user_confirmed_contradicting_confirmation_flag(self):
        raw = valid_proposal_dict()
        raw["length"]["source"] = "user_confirmed"
        assert_rejected(self, raw, "length", phase="plausibility")


class TestConfirmationRejections(unittest.TestCase):
    def test_unlisted_null_thickness(self):
        raw = valid_proposal_dict()
        raw["requires_confirmation"] = [c for c in raw["requires_confirmation"]
                                        if c != "thickness"]
        assert_rejected(self, raw, "thickness", phase="confirmation")

    def test_unlisted_ungrounded_scale(self):
        raw = valid_proposal_dict()
        raw["requires_confirmation"] = [c for c in raw["requires_confirmation"]
                                        if c != "scale"]
        assert_rejected(self, raw, "scale", phase="confirmation")

    def test_listed_but_unflagged_field(self):
        raw = valid_proposal_dict()
        raw["length"]["requires_confirmation"] = False
        assert_rejected(self, raw, "length", phase="confirmation")

    def test_listed_nonexistent_path(self):
        raw = valid_proposal_dict()
        raw["requires_confirmation"].append("holes[9].diameter")
        assert_rejected(self, raw, "holes[9].diameter", phase="confirmation")

    def test_null_depth_requires_hole_confirmation(self):
        raw = valid_proposal_dict()
        raw["holes"][0]["depth"] = None
        raw["requires_confirmation"].append("holes[0]")
        validate_proposal(raw)  # listed: accepted
        raw["requires_confirmation"].remove("holes[0]")
        assert_rejected(self, raw, "holes[0]", phase="confirmation")


class TestJsonParsing(unittest.TestCase):
    def test_valid_json_round_trip(self):
        import json

        proposal = parse_proposal_json(json.dumps(valid_proposal_dict()))
        self.assertEqual(len(proposal.holes), 4)

    def test_malformed_json_rejected(self):
        with self.assertRaises(ProposalError) as ctx:
            parse_proposal_json('{"classification": ')
        self.assertIn("structural", str(ctx.exception))

    def test_nan_literal_rejected(self):
        with self.assertRaises(ProposalError) as ctx:
            parse_proposal_json('{"value": NaN}')
        self.assertIn("non-finite", str(ctx.exception))

    def test_non_string_input_rejected(self):
        with self.assertRaises(ProposalError):
            parse_proposal_json(None)

    def test_issues_carry_phases(self):
        raw = {"classification": "engine_block", "units": "cm"}
        with self.assertRaises(ProposalError) as ctx:
            validate_proposal(raw)
        phases = {issue.phase for issue in ctx.exception.issues}
        self.assertTrue(phases <= {"structural", "plausibility", "confirmation"})
        self.assertIn("structural", phases)
        fields = {issue.field for issue in ctx.exception.issues}
        self.assertTrue({"classification", "units"} <= fields)


if __name__ == "__main__":
    unittest.main()
