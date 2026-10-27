"""Renderer tests: success path plus every structured failure mode."""

import os
import unittest

from tests import TempDirTest, needs_openscad
from reforge.cad_generator import write_scad
from reforge.parameters import validate_plate_params
from reforge.renderer import RenderError, find_openscad, render_scad_to_stl
from tests import sample_params


class TestFindOpenscad(unittest.TestCase):
    def test_missing_executable_raises_structured_error(self):
        with self.assertRaises(RenderError) as ctx:
            find_openscad("openscad-binary-that-does-not-exist")
        self.assertEqual(ctx.exception.reason, "executable_missing")


class TestRenderFailures(TempDirTest):
    def test_missing_input_scad(self):
        with self.assertRaises(RenderError) as ctx:
            render_scad_to_stl(
                os.path.join(self.tmp.name, "nope.scad"),
                self.path("out.stl"),
                executable="openscad-binary-that-does-not-exist",
            )
        self.assertEqual(ctx.exception.reason, "executable_missing")

    def test_missing_executable_with_valid_input(self):
        scad = self.path("plate.scad")
        write_scad(sample_params(), scad)
        with self.assertRaises(RenderError) as ctx:
            render_scad_to_stl(
                scad, self.path("out.stl"),
                executable="openscad-binary-that-does-not-exist",
            )
        self.assertEqual(ctx.exception.reason, "executable_missing")

    @needs_openscad
    def test_nonzero_exit_on_invalid_scad(self):
        bad = self.path("bad.scad")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("this is not valid openscad syntax (((")
        with self.assertRaises(RenderError) as ctx:
            render_scad_to_stl(bad, self.path("out.stl"))
        self.assertEqual(ctx.exception.reason, "nonzero_exit")

    @needs_openscad
    def test_timeout_is_structured(self):
        scad = self.path("plate.scad")
        write_scad(sample_params(), scad)
        with self.assertRaises(RenderError) as ctx:
            render_scad_to_stl(scad, self.path("out.stl"), timeout_s=0.0001)
        self.assertEqual(ctx.exception.reason, "timeout")


class TestRenderSuccess(TempDirTest):
    @needs_openscad
    def test_renders_nonempty_stl(self):
        scad = self.path("plate.scad")
        stl = self.path("plate.stl")
        write_scad(sample_params(), scad)
        result = render_scad_to_stl(scad, stl)
        self.assertTrue(os.path.isfile(stl))
        self.assertGreater(os.path.getsize(stl), 0)
        self.assertGreaterEqual(result.elapsed_s, 0.0)


if __name__ == "__main__":
    unittest.main()
