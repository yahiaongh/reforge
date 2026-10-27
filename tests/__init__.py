"""Shared helpers for the Reforge test suite (stdlib only)."""

import os
import shutil
import sys
import tempfile
import unittest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from reforge.parameters import validate_plate_params  # noqa: E402

OPENSCAD = shutil.which("openscad")

SAMPLE_KWARGS = dict(
    length=60.0,
    width=40.0,
    thickness=5.0,
    holes=[(10, 10), (50, 10), (10, 30), (50, 30)],
    hole_diameter=6.0,
)


def sample_params():
    return validate_plate_params(**SAMPLE_KWARGS)


def needs_openscad(test):
    return unittest.skipIf(OPENSCAD is None, "OpenSCAD executable not found")(test)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="reforge-test-")
        self.addCleanup(self.tmp.cleanup)

    def path(self, name: str) -> str:
        return os.path.join(self.tmp.name, name)
