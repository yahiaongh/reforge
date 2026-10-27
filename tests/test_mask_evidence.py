"""Mask-evidence extraction tests. Fully offline: synthetic fixtures only,
no network, no models, no new dependencies."""

import unittest

import numpy as np

from reforge.mask_evidence import (
    MaskEvidenceError,
    extract_mask_evidence,
    to_proposal_evidence,
)

try:
    from test_geometry import valid_proposal_dict
except ImportError:  # run as tests.test_mask_evidence instead of discover
    from tests.test_geometry import valid_proposal_dict


IMAGE_SIZE = (100, 80)
GOOD_HASH = "ab" * 32


def summary_for(raster, bounds=None, confidence=0.9, extra_masks=0):
    height, width = raster.shape
    if bounds is None:
        bounds = {"left": 10.0, "top": 10.0,
                  "right": 10.0 + width, "bottom": 10.0 + height}
    mask = {"kind": "mask", "object_id": "o1", "bounds": bounds,
            "mask": {"encoding": "one_bit", "payload": "<fixture>",
                     "width": width, "height": height},
            "confidence": confidence}
    masks = [mask] + [
        {"kind": "mask", "object_id": f"o{i}", "bounds": bounds,
         "mask": {"encoding": "one_bit", "payload": "<fixture>",
                  "width": width, "height": height},
         "confidence": confidence}
        for i in range(1, extra_masks + 1)]
    return {"v": 1, "model": "sam-3.1", "concept": "flat metal plate",
            "asset_hash": GOOD_HASH, "boxes": [], "masks": masks,
            "diagnostics": [], "outcome": "completed"}


def decode_for(raster):
    blob = raster.astype(np.uint8).tobytes()

    def decode(mask):
        return blob

    return decode


def block_with_hole(outer=12, hole=4):
    raster = np.ones((outer, outer), dtype=bool)
    start = (outer - hole) // 2
    raster[start:start + hole, start:start + hole] = False
    return raster


class TestNormalMasks(unittest.TestCase):
    def test_block_with_hole_reports_gap(self):
        raster = block_with_hole()
        evidence = extract_mask_evidence(
            summary_for(raster), IMAGE_SIZE, decode_fn=decode_for(raster))
        self.assertEqual(evidence.interior_gap_count, 1)
        self.assertEqual(list(evidence.interior_gap_sizes_px), [16])
        self.assertAlmostEqual(evidence.fill_fraction, (144 - 16) / 144)
        self.assertGreater(evidence.boundary_pixel_count, 0)
        self.assertTrue(any("interior_gaps_detected" in w for w in evidence.warnings))

    def test_solid_block_reports_no_gaps_conservatively(self):
        raster = np.ones((10, 10), dtype=bool)
        evidence = extract_mask_evidence(
            summary_for(raster), IMAGE_SIZE, decode_fn=decode_for(raster))
        self.assertEqual(evidence.interior_gap_count, 0)
        # Zero gaps must never read as proof of solidity.
        self.assertTrue(any("does not establish" in w for w in evidence.warnings))

    def test_border_touching_gap_is_exterior(self):
        raster = np.ones((10, 10), dtype=bool)
        raster[0:3, 0:3] = False  # notch open to the raster border
        evidence = extract_mask_evidence(
            summary_for(raster), IMAGE_SIZE, decode_fn=decode_for(raster))
        self.assertEqual(evidence.interior_gap_count, 0)

    def test_quantization_recorded(self):
        raster = np.ones((10, 20), dtype=bool)
        bounds = {"left": 0.0, "top": 0.0, "right": 40.0, "bottom": 10.0}
        evidence = extract_mask_evidence(
            summary_for(raster, bounds=bounds), IMAGE_SIZE,
            decode_fn=decode_for(raster))
        self.assertEqual(evidence.quant_px_per_cell, (2.0, 1.0))

    def test_multiple_masks_analyzes_first_with_warning(self):
        raster = np.ones((10, 10), dtype=bool)
        evidence = extract_mask_evidence(
            summary_for(raster, extra_masks=1), IMAGE_SIZE,
            decode_fn=decode_for(raster))
        self.assertEqual(evidence.mask_count, 2)
        self.assertEqual(evidence.mask_index, 0)
        self.assertTrue(any("2 masks" in w for w in evidence.warnings))


class TestDegenerateMasks(unittest.TestCase):
    def test_empty_mask_warns_explicitly(self):
        raster = np.zeros((10, 10), dtype=bool)
        evidence = extract_mask_evidence(
            summary_for(raster), IMAGE_SIZE, decode_fn=decode_for(raster))
        self.assertEqual(evidence.fill_fraction, 0.0)
        self.assertEqual(evidence.boundary_pixel_count, 0)
        self.assertTrue(any(w.startswith("empty_mask") for w in evidence.warnings))

    def test_full_frame_mask_warns_explicitly(self):
        raster = np.ones((80, 100), dtype=bool)
        bounds = {"left": 0.0, "top": 0.0, "right": 100.0, "bottom": 80.0}
        evidence = extract_mask_evidence(
            summary_for(raster, bounds=bounds), IMAGE_SIZE,
            decode_fn=decode_for(raster))
        self.assertTrue(any(w.startswith("full_frame_mask") for w in evidence.warnings))

    def test_tiny_raster_warns_degenerate(self):
        raster = np.ones((2, 2), dtype=bool)
        evidence = extract_mask_evidence(
            summary_for(raster), IMAGE_SIZE, decode_fn=decode_for(raster))
        self.assertTrue(any("degenerate_geometry_tiny" in w for w in evidence.warnings))

    def test_bounds_outside_image_warns(self):
        raster = np.ones((10, 10), dtype=bool)
        bounds = {"left": -5.0, "top": 0.0, "right": 50.0, "bottom": 10.0}
        evidence = extract_mask_evidence(
            summary_for(raster, bounds=bounds), IMAGE_SIZE,
            decode_fn=decode_for(raster))
        self.assertTrue(any("bounds_exceed_image" in w for w in evidence.warnings))


class TestMalformedMasks(unittest.TestCase):
    def test_non_mapping_summary(self):
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence([1, 2], IMAGE_SIZE, decode_fn=lambda m: b"")

    def test_wrong_version(self):
        summary = summary_for(np.ones((4, 4), dtype=bool))
        summary["v"] = 2
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary, IMAGE_SIZE, decode_fn=lambda m: b"")

    def test_no_masks(self):
        summary = summary_for(np.ones((4, 4), dtype=bool))
        summary["masks"] = []
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary, IMAGE_SIZE, decode_fn=lambda m: b"")

    def test_degenerate_bounds(self):
        raster = np.ones((4, 4), dtype=bool)
        bounds = {"left": 5.0, "top": 0.0, "right": 5.0, "bottom": 4.0}
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary_for(raster, bounds=bounds), IMAGE_SIZE,
                                  decode_fn=decode_for(raster))

    def test_bad_raster_dimensions(self):
        summary = summary_for(np.ones((4, 4), dtype=bool))
        summary["masks"][0]["mask"]["width"] = 0
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary, IMAGE_SIZE, decode_fn=lambda m: b"")

    def test_undecodable_payload(self):
        def bad_decode(mask):
            raise ValueError("bad base85")

        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary_for(np.ones((4, 4), dtype=bool)),
                                  IMAGE_SIZE, decode_fn=bad_decode)

    def test_short_payload(self):
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary_for(np.ones((4, 4), dtype=bool)),
                                  IMAGE_SIZE, decode_fn=lambda m: b"\x01\x00")

    def test_non_binary_raster(self):
        blob = (np.ones((4, 4), dtype=np.uint8) * 2).tobytes()
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary_for(np.ones((4, 4), dtype=bool)),
                                  IMAGE_SIZE, decode_fn=lambda m: blob)

    def test_bad_image_size(self):
        with self.assertRaises(MaskEvidenceError):
            extract_mask_evidence(summary_for(np.ones((4, 4), dtype=bool)),
                                  (100, -80), decode_fn=lambda m: b"")


class TestProposalConformance(unittest.TestCase):
    def test_evidence_record_fits_geometry_schema(self):
        from reforge.geometry import validate_proposal

        raster = block_with_hole()
        evidence = extract_mask_evidence(
            summary_for(raster), IMAGE_SIZE, decode_fn=decode_for(raster))
        record = to_proposal_evidence(evidence)
        self.assertEqual(set(record),
                         {"image_asset_hash", "mask_bounds", "box_confidence", "notes"})
        self.assertLessEqual(len(record["notes"]), 500)
        self.assertIn("pixels", record["notes"])
        proposal = valid_proposal_dict()
        proposal["evidence"] = record
        self.assertIsNotNone(validate_proposal(proposal))

    def test_conflicting_hash_reported_not_reshaped(self):
        raster = np.ones((10, 10), dtype=bool)
        evidence = extract_mask_evidence(
            summary_for(raster), IMAGE_SIZE, decode_fn=decode_for(raster))
        tampered = evidence.__class__(**{**evidence.__dict__, "asset_hash": "xyz"})
        with self.assertRaises(MaskEvidenceError):
            to_proposal_evidence(tampered)

    def test_out_of_range_confidence_reported(self):
        raster = np.ones((10, 10), dtype=bool)
        summary = summary_for(raster, confidence=7.0)
        evidence = extract_mask_evidence(
            summary, IMAGE_SIZE, decode_fn=decode_for(raster))
        with self.assertRaises(MaskEvidenceError):
            to_proposal_evidence(evidence)


if __name__ == "__main__":
    unittest.main()
