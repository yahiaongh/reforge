"""Independent geometric validation of exported STL files.

The validator parses the STL on its own (ASCII and binary) with only NumPy,
then measures the mesh directly:

- nonempty geometry, finite coordinates, degenerate-face fraction;
- closedness via an undirected edge-sharing count (informational, see below);
- bounding-box dimensions against the expected parameters;
- per-hole through-hole presence, centre, and diameter measured by casting
  vertical rays through the mesh (Moller-Trumbore), NOT by trusting the
  generator's inputs.

Scope limits, stated plainly: the ray tests verify through-holes whose axes
are perpendicular to the plate (the only holes this pipeline generates). They
do not prove the absence of extra unintended holes elsewhere; the bounding
box plus manifold checks constrain that risk but do not eliminate it. Blind
holes are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import struct
from typing import Any

import numpy as np

from .parameters import PlateParams


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    measured: Any = None
    expected: Any = None


@dataclass
class ValidationReport:
    stl_path: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    facet_count: int = 0

    def to_dict(self) -> dict:
        return {
            "stl_path": self.stl_path,
            "passed": self.passed,
            "facet_count": self.facet_count,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "measured": c.measured,
                    "expected": c.expected,
                }
                for c in self.checks
            ],
        }

    def human_summary(self) -> str:
        lines = [
            f"Validation of {self.stl_path}: {'PASS' if self.passed else 'FAIL'} "
            f"({self.facet_count} facets, {len(self.checks)} checks)"
        ]
        for check in self.checks:
            mark = "ok  " if check.passed else "FAIL"
            lines.append(f"  [{mark}] {check.name}: {check.detail}")
        return "\n".join(lines)


def read_stl_triangles(path: str) -> np.ndarray:
    """Parse an STL file (ASCII or binary) into an (N, 3, 3) float64 array."""
    with open(path, "rb") as handle:
        head = handle.read(84)
        if len(head) < 84:
            raise ValueError(f"file too small to be an STL: {path}")
        handle.seek(0)
        text_start = head[:256].decode("ascii", errors="replace").lstrip().lower()
        if text_start.startswith("solid"):
            # Could still be a binary STL whose header starts with "solid";
            # a binary file has a plausible facet count at byte 80.
            (count,) = struct.unpack("<I", head[80:84])
            size = os.path.getsize(path)
            if count > 0 and 84 + count * 50 == size:
                return _read_binary_stl(path, count)
            return _read_ascii_stl(path)
        (count,) = struct.unpack("<I", head[80:84])
        return _read_binary_stl(path, count)


def _read_ascii_stl(path: str) -> np.ndarray:
    vertices: list[list[float]] = []
    with open(path, "r", encoding="ascii", errors="strict") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError as exc:
                    raise ValueError(f"malformed vertex line: {line.strip()!r}") from exc
    if len(vertices) % 3 != 0:
        raise ValueError(f"ASCII STL vertex count {len(vertices)} is not a multiple of 3")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _read_binary_stl(path: str, count: int) -> np.ndarray:
    size = os.path.getsize(path)
    if size != 84 + count * 50:
        raise ValueError(
            f"binary STL size mismatch: header claims {count} facets "
            f"but file size is {size} bytes"
        )
    with open(path, "rb") as handle:
        handle.seek(84)
        blob = handle.read()
    data = np.frombuffer(blob, dtype=np.uint8).reshape(count, 50)
    floats = data[:, :48].view("<f4").reshape(count, 4, 3).astype(np.float64)
    return floats[:, 1:, :]  # skip the normal, keep the 3 vertices


def _ray_hits(triangles: np.ndarray, x: float, y: float, z_top: float) -> int:
    """Count intersections of the downward ray (x, y, z_top+something) -> -z.

    Uses Moller-Trumbore over all triangles. Meshes from this pipeline have a
    few hundred facets, so the brute-force loop is fast (<1 ms per ray).
    """
    origin = np.array([x, y, z_top], dtype=np.float64)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    eps = 1e-9
    v0 = triangles[:, 0, :]
    e1 = triangles[:, 1, :] - v0
    e2 = triangles[:, 2, :] - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    valid = np.abs(det) > eps
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,j->i", qvec, direction) * inv_det
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    hits = valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
    return int(np.count_nonzero(hits))


#: Deterministic sub-facet ray offsets (mm). A ray cast exactly along a mesh
#: symmetry line can graze a shared triangle edge and be counted twice.
#: Grazing is a measure-zero event, so a majority vote over these tiny offsets
#: classifies the column robustly while moving the query point by ~0.001 mm
#: (far below any tolerance).
_RAY_JITTER = ((0.0, 0.0), (0.0007, 0.0013), (-0.0011, 0.0009))


def _is_solid(triangles: np.ndarray, x: float, y: float, z_top: float) -> bool:
    """True when the vertical column at (x, y) contains solid material.

    The ray starts above the part and points down: a solid column crosses the
    top face and the bottom face (2 hits, more if a ray grazes a shared edge),
    while a through-hole column hits nothing. Majority vote over jittered rays
    so edge-grazing double counts cannot flip the verdict.
    """
    votes = sum(
        _ray_hits(triangles, x + dx, y + dy, z_top) >= 2 for dx, dy in _RAY_JITTER
    )
    return votes >= 2


def _measure_hole(
    triangles: np.ndarray, cx: float, cy: float, nominal_d: float, z_top: float
) -> tuple[float, float, float]:
    """Measure a through-hole by ray sweeps along +/-x and +/-y.

    Returns (measured_cx, measured_cy, measured_diameter). Raises ValueError if
    no hole is found at the expected centre (the column is solid).
    """
    if _is_solid(triangles, cx, cy, z_top):
        raise ValueError(f"column at ({cx}, {cy}) is solid; expected a through-hole")
    step = 0.1
    limit = nominal_d / 2.0 + 2.0

    def edge(sx: float, sy: float) -> float:
        dist = 0.0
        while dist <= limit:
            dist += step
            if _is_solid(triangles, cx + sx * dist, cy + sy * dist, z_top):
                return dist - step / 2.0
        raise ValueError(f"hole at ({cx}, {cy}) has no wall within {limit} mm")

    xp, xn = edge(1.0, 0.0), edge(-1.0, 0.0)
    yp, yn = edge(0.0, 1.0), edge(0.0, -1.0)
    measured_cx = cx + (xp - xn) / 2.0
    measured_cy = cy + (yp - yn) / 2.0
    measured_d = (xp + xn + yp + yn) / 2.0
    return measured_cx, measured_cy, measured_d


def validate_stl(
    stl_path: str,
    params: PlateParams,
    bbox_tol: float = 0.25,
    hole_diam_tol: float = 0.6,
    hole_pos_tol: float = 0.6,
    max_degenerate_fraction: float = 0.01,
) -> ValidationReport:
    """Validate *stl_path* against *params* and return a machine-readable report."""
    report = ValidationReport(stl_path=stl_path, passed=False)

    def add(name: str, passed: bool, detail: str, measured=None, expected=None) -> None:
        report.checks.append(
            Check(name=name, passed=passed, detail=detail, measured=measured, expected=expected)
        )

    if not os.path.isfile(stl_path):
        add("file_present", False, f"STL file not found: {stl_path}")
        return report
    if os.path.getsize(stl_path) == 0:
        add("file_present", False, "STL file is empty")
        return report
    add("file_present", True, "STL file exists and is nonempty")

    try:
        triangles = read_stl_triangles(stl_path)
    except (ValueError, struct.error, UnicodeDecodeError, OSError) as exc:
        add("stl_parseable", False, f"STL parsing failed: {exc}")
        return report
    report.facet_count = int(triangles.shape[0])
    add("stl_parseable", True, f"parsed {report.facet_count} facets")

    if triangles.shape[0] == 0:
        add("nonempty_geometry", False, "mesh has zero facets")
        return report
    add("nonempty_geometry", True, f"mesh has {triangles.shape[0]} facets")

    if not np.all(np.isfinite(triangles)):
        add("finite_coordinates", False, "mesh contains NaN or infinite coordinates")
        return report
    add("finite_coordinates", True, "all vertex coordinates are finite")

    edge1 = triangles[:, 1, :] - triangles[:, 0, :]
    edge2 = triangles[:, 2, :] - triangles[:, 0, :]
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    degenerate = int(np.count_nonzero(areas <= 1e-12))
    degenerate_fraction = degenerate / max(1, triangles.shape[0])
    add(
        "degenerate_faces",
        degenerate_fraction <= max_degenerate_fraction,
        f"{degenerate}/{triangles.shape[0]} degenerate faces "
        f"(limit {max_degenerate_fraction:.1%})",
        measured=degenerate,
    )

    # Closedness: every undirected edge of a closed solid is shared by exactly
    # two facets. Reported as informational context for extra-material risk;
    # it gates the report because an open mesh cannot be trusted for the
    # bounding-box and ray measurements below.
    rounded = np.round(triangles, decimals=6)
    edges = np.concatenate(
        [
            np.sort(np.stack([rounded[:, 0, :], rounded[:, 1, :]], axis=1), axis=1),
            np.sort(np.stack([rounded[:, 1, :], rounded[:, 2, :]], axis=1), axis=1),
            np.sort(np.stack([rounded[:, 2, :], rounded[:, 0, :]], axis=1), axis=1),
        ],
        axis=0,
    )
    _, counts = np.unique(edges.reshape(-1, 6), axis=0, return_counts=True)
    bad_edges = int(np.count_nonzero(counts != 2))
    add(
        "closed_manifold_edges",
        bad_edges == 0,
        f"{bad_edges} edges not shared by exactly 2 facets",
        measured=bad_edges,
        expected=0,
    )

    lo = triangles.min(axis=(0, 1))
    hi = triangles.max(axis=(0, 1))
    measured_dims = [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])]
    expected_dims = [params.length, params.width, params.thickness]
    for axis, (meas, exp) in enumerate(zip(measured_dims, expected_dims)):
        ok = abs(meas - exp) <= bbox_tol
        add(
            f"bbox_axis_{'XYZ'[axis]}",
            ok,
            f"extent {meas:.3f} mm vs expected {exp:.3f} mm (tol {bbox_tol} mm)",
            measured=round(meas, 4),
            expected=exp,
        )

    z_top = float(hi[2]) + 1.0
    for index, hole in enumerate(params.holes):
        name = f"hole_{index}"
        try:
            mcx, mcy, md = _measure_hole(triangles, hole.x, hole.y, hole.diameter, z_top)
        except ValueError as exc:
            add(name, False, str(exc), expected={"x": hole.x, "y": hole.y, "diameter": hole.diameter})
            continue
        pos_err = float(max(abs(mcx - hole.x), abs(mcy - hole.y)))
        diam_err = abs(md - hole.diameter)
        ok = pos_err <= hole_pos_tol and diam_err <= hole_diam_tol
        add(
            name,
            ok,
            f"centre ({mcx:.2f}, {mcy:.2f}) vs ({hole.x}, {hole.y}); "
            f"diameter {md:.2f} vs {hole.diameter} "
            f"(pos err {pos_err:.2f}, diam err {diam_err:.2f})",
            measured={"x": round(mcx, 3), "y": round(mcy, 3), "diameter": round(md, 3)},
            expected={"x": hole.x, "y": hole.y, "diameter": hole.diameter},
        )

    report.passed = all(c.passed for c in report.checks)
    return report
