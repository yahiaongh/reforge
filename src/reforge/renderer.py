"""Local STL export through the installed OpenSCAD CLI.

No inference is involved. All failure modes (missing executable, process
failure, timeout, missing/invalid output) are reported as structured
:class:`RenderError` values, never as exceptions leaking platform detail to
callers that need machine-readable reports.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import time


class RenderError(RuntimeError):
    """Structured rendering failure with a machine-readable ``reason`` code."""

    #: executable_missing | timeout | nonzero_exit | output_missing | output_empty
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class RenderResult:
    stl_path: str
    elapsed_s: float
    stdout: str = ""
    stderr: str = ""


def find_openscad(executable: str = "openscad") -> str:
    """Resolve the OpenSCAD executable or raise a structured error."""
    if os.path.isabs(executable) or os.sep in executable:
        if os.path.isfile(executable) and os.access(executable, os.X_OK):
            return executable
    else:
        resolved = shutil.which(executable)
        if resolved:
            return resolved
    raise RenderError("executable_missing", f"OpenSCAD executable not found: {executable!r}")


def render_scad_to_stl(
    scad_path: str,
    stl_path: str,
    executable: str = "openscad",
    timeout_s: float = 180.0,
) -> RenderResult:
    """Export *scad_path* to *stl_path* via ``openscad -o``."""
    binary = find_openscad(executable)
    if not os.path.isfile(scad_path):
        raise RenderError("output_missing", f"input .scad file not found: {scad_path}")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [binary, "-o", stl_path, scad_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise RenderError("executable_missing", f"OpenSCAD not executable: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError(
            "timeout", f"OpenSCAD timed out after {timeout_s:g}s rendering {scad_path}"
        ) from exc
    elapsed = time.monotonic() - started
    stderr = completed.stderr or ""
    stdout = completed.stdout or ""
    if completed.returncode != 0:
        raise RenderError(
            "nonzero_exit",
            f"OpenSCAD exited with status {completed.returncode}: {stderr.strip()[-2000:]}",
        )
    if not os.path.isfile(stl_path):
        raise RenderError("output_missing", f"OpenSCAD produced no output file: {stl_path}")
    if os.path.getsize(stl_path) == 0:
        raise RenderError("output_empty", f"OpenSCAD produced an empty STL: {stl_path}")
    return RenderResult(stl_path=stl_path, elapsed_s=elapsed, stdout=stdout, stderr=stderr)
