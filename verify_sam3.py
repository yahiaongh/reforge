"""Legacy SAM segmentation entry point — now a thin wrapper.

All requests go through :mod:`reforge.sam_experiment`, which enforces the
budget guard in-process (at most one paid request, no retries, every outcome
ledgered). This script performs no API call itself.

Usage: python verify_sam3.py <image_path> <noun_phrase> [out.png]
       plus optional: --experiment-id ID --ledger PATH --cache-dir DIR --max-cost USD
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from reforge.sam_experiment import run_sam_experiment


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            flags[argv[i][2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    if len(args) < 3:
        sys.exit("Usage: python verify_sam3.py <image_path> <noun_phrase> [out.png]")
    out = args[3] if len(args) > 3 else "assets/overlay.png"
    max_cost = float(flags["max_cost"]) if "max_cost" in flags else None
    result = run_sam_experiment(
        image_path=args[1],
        concept=args[2],
        out_path=out,
        experiment_id=flags.get("experiment_id", "sam-mechanical-001"),
        ledger_path=flags.get("ledger"),
        cache_dir=flags.get("cache_dir"),
        max_cost_usd=max_cost,
    )
    print(f"[{result.status}] {result.detail}")
    if result.status in ("ok", "cached"):
        print(f"boxes={result.box_count} masks={result.mask_count} overlay={result.overlay_path}")
        return 0
    return 3 if result.status == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
