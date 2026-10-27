"""End-to-end test: parameters -> SCAD -> STL -> independent validation."""

import json
import os
import unittest

from tests import TempDirTest, needs_openscad, sample_params
from reforge.cad_generator import write_scad
from reforge.renderer import render_scad_to_stl
from reforge.validator import validate_stl


class TestEndToEndLocalPipeline(TempDirTest):
    @needs_openscad
    def test_params_to_validated_stl(self):
        params = sample_params()
        scad = self.path("plate.scad")
        stl = self.path("plate.stl")
        report_json = self.path("report.json")

        write_scad(params, scad)
        self.assertTrue(os.path.isfile(scad))

        result = render_scad_to_stl(scad, stl)
        self.assertTrue(os.path.isfile(result.stl_path))

        report = validate_stl(stl, params)
        with open(report_json, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)

        self.assertTrue(report.passed, msg=report.human_summary())
        # The STL must open in OpenSCAD: re-render it to a second STL via
        # an import wrapper as a smoke check that the file is consumable.
        wrapper = self.path("wrapper.scad")
        with open(wrapper, "w", encoding="utf-8") as handle:
            handle.write(f'import("{stl}");\n')
        render_scad_to_stl(wrapper, self.path("reimport.stl"))
        self.assertTrue(os.path.isfile(self.path("reimport.stl")))


if __name__ == "__main__":
    unittest.main()
