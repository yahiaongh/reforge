"""Command-line interface: generate, render, validate, and run the pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from reforge.cad_generator import generate_scad, write_scad
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from reforge.cad_generator import generate_scad, write_scad

from reforge.parameters import (
    ParameterError,
    PlateParams,
    validate_plate_params,
)
from reforge.renderer import RenderError, render_scad_to_stl
from reforge.validator import validate_stl


def parse_holes(spec: str | None, hole_diameter: float | None) -> list:
    """Parse ``--holes "x1,y1[,(d1)];x2,y2[,(d2)]"`` into hole specs."""
    if not spec:
        return []
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) == 2 and hole_diameter is not None:
            out.append((float(parts[0]), float(parts[1])))
        elif len(parts) == 3:
            out.append((float(parts[0]), float(parts[1]), float(parts[2])))
        else:
            raise ParameterError(
                f"cannot parse hole {chunk!r}: use 'x,y' (with --hole-diameter) or 'x,y,d'"
            )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reforge", description="Local parametric CAD pipeline for planar parts"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="validate params and write .scad (and .stl)")
    gen.add_argument("--length", type=float, required=True)
    gen.add_argument("--width", type=float, required=True)
    gen.add_argument("--thickness", type=float, required=True)
    gen.add_argument("--hole-diameter", type=float, default=None)
    gen.add_argument("--holes", default=None, help="'x1,y1;x2,y2' or 'x1,y1,d1;...'")
    gen.add_argument("--scad", required=True, help="output .scad path")
    gen.add_argument("--stl", default=None, help="also render this .stl path")
    gen.add_argument("--openscad", default="openscad")
    gen.add_argument("--timeout", type=float, default=180.0)

    val = sub.add_parser("validate", help="validate an STL against parameters")
    val.add_argument("--stl", required=True)
    val.add_argument("--length", type=float, required=True)
    val.add_argument("--width", type=float, required=True)
    val.add_argument("--thickness", type=float, required=True)
    val.add_argument("--hole-diameter", type=float, default=None)
    val.add_argument("--holes", default=None)
    val.add_argument("--report", default=None, help="write JSON report here")
    val.add_argument("--bbox-tol", type=float, default=0.25)
    val.add_argument("--hole-diam-tol", type=float, default=0.6)
    val.add_argument("--hole-pos-tol", type=float, default=0.6)

    pipe = sub.add_parser("pipeline", help="params -> scad -> stl -> validation report")
    pipe.add_argument("--length", type=float, required=True)
    pipe.add_argument("--width", type=float, required=True)
    pipe.add_argument("--thickness", type=float, required=True)
    pipe.add_argument("--hole-diameter", type=float, default=None)
    pipe.add_argument("--holes", default=None)
    pipe.add_argument("--outdir", required=True)
    pipe.add_argument("--openscad", default="openscad")
    pipe.add_argument("--timeout", type=float, default=180.0)
    pipe.add_argument(
        "--constraints-json", default=None,
        help="JSON file with user-known measurements to validate against; "
             "defaults to the candidate parameters themselves (self-check)",
    )
    pipe.add_argument(
        "--correct", action="store_true",
        help="run bounded deterministic correction when validation fails",
    )
    pipe.add_argument(
        "--adjustable", default="holes",
        help="comma-separated fields the correction agent may change "
             "(subset of length,width,thickness,holes); all others are "
             "treated as user-known and immutable",
    )
    pipe.add_argument("--max-corrections", type=int, default=2)

    sam = sub.add_parser(
        "sam-segment",
        help="guarded single-shot SAM 3.1 segmentation (exactly one paid request max)",
    )
    sam.add_argument("--image", required=True, help="input photograph path")
    sam.add_argument("--concept", required=True, help="segmentation noun phrase")
    sam.add_argument("--out", required=True, help="overlay PNG output path")
    sam.add_argument("--experiment-id", default="sam-mechanical-001")
    sam.add_argument("--ledger", default=None, help="ledger path (default ./ledger/usage.jsonl)")
    sam.add_argument("--cache-dir", default=None, help="cache dir (default <ledger-dir>/cache)")
    sam.add_argument(
        "--max-cost", type=float, default=None,
        help="operator-declared policy bound in USD (default: per-experiment cap). "
             "Local policy only, not a provider-side cap.",
    )

    probe = sub.add_parser(
        "probe-dry-run",
        help="offline evidence-to-request dry run (never transmits)",
    )
    probe.add_argument("--image", required=True, help="input photograph path")
    probe.add_argument("--mask-summary", default=None,
                       help="cached SAM mask-summary JSON path (optional)")
    probe.add_argument("--concept", default="flat metal plate with circular holes")
    probe.add_argument("--classification", default="rectangular_plate")
    probe.add_argument("--model", default=None,
                       help="model id (default: operator-supplied Spark default)")
    probe.add_argument("--max-cost", type=float, default=None)
    probe.add_argument("--ledger", default=None)

    bprobe = sub.add_parser(
        "baseten-dry-run",
        help="offline Baseten Chat Completions dry run (never transmits)",
    )
    bprobe.add_argument("--image", required=True, help="input photograph path")
    bprobe.add_argument("--mask-summary", default=None,
                        help="cached SAM mask-summary JSON path (optional)")
    bprobe.add_argument("--concept", default="flat metal plate with circular holes")
    bprobe.add_argument("--classification", default="rectangular_plate")
    bprobe.add_argument("--model", default=None,
                        help="model slug (default: meta-models/muse-spark-open-1026)")
    bprobe.add_argument("--max-tokens", type=int, default=1500,
                        help="output-length cap only, not a dollar cap")
    bprobe.add_argument("--max-cost", type=float, default=None)
    bprobe.add_argument("--ledger", default=None)

    budget = sub.add_parser("budget", help="show ledger totals and discretionary budget")
    budget.add_argument("--ledger", default=None, help="ledger path (default ./ledger/usage.jsonl)")
    budget.add_argument("--limit", type=float, default=150.0)
    budget.add_argument("--reserve", type=float, default=25.0)
    budget.add_argument("--per-experiment-cap", type=float, default=5.0)
    return parser


def params_from_args(args) -> PlateParams:
    return validate_plate_params(
        length=args.length,
        width=args.width,
        thickness=args.thickness,
        holes=parse_holes(args.holes, args.hole_diameter),
        hole_diameter=args.hole_diameter,
    )


def cmd_generate(args) -> int:
    try:
        params = params_from_args(args)
    except ParameterError as exc:
        print(f"error: invalid parameters: {exc}", file=sys.stderr)
        return 2
    write_scad(params, args.scad)
    print(f"wrote {args.scad}")
    if args.stl:
        try:
            result = render_scad_to_stl(args.scad, args.stl, args.openscad, args.timeout)
        except RenderError as exc:
            print(f"error: render failed [{exc.reason}]: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {result.stl_path} in {result.elapsed_s:.1f}s")
    return 0


def cmd_validate(args) -> int:
    try:
        params = params_from_args(args)
    except ParameterError as exc:
        print(f"error: invalid parameters: {exc}", file=sys.stderr)
        return 2
    report = validate_stl(
        args.stl,
        params,
        bbox_tol=args.bbox_tol,
        hole_diam_tol=args.hole_diam_tol,
        hole_pos_tol=args.hole_pos_tol,
    )
    print(report.human_summary())
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        print(f"wrote {args.report}")
    return 0 if report.passed else 1


def load_constraints(args) -> PlateParams:
    from reforge.parameters import plate_params_from_dict

    if not args.constraints_json:
        return params_from_args(args)
    with open(args.constraints_json, encoding="utf-8") as handle:
        return plate_params_from_dict(json.load(handle))


def cmd_pipeline(args) -> int:
    from reforge.correction_agent import CorrectionAgent

    try:
        params = params_from_args(args)
    except ParameterError as exc:
        print(f"error: invalid parameters: {exc}", file=sys.stderr)
        return 2
    try:
        constraints = load_constraints(args)
    except (ParameterError, OSError, ValueError) as exc:
        print(f"error: invalid constraints file: {exc}", file=sys.stderr)
        return 2
    os.makedirs(args.outdir, exist_ok=True)
    scad_path = os.path.join(args.outdir, "plate.scad")
    stl_path = os.path.join(args.outdir, "plate.stl")
    report_path = os.path.join(args.outdir, "validation_report.json")
    write_scad(params, scad_path)
    print(f"wrote {scad_path}")

    def rebuild(candidate: PlateParams):
        write_scad(candidate, scad_path)
        render_scad_to_stl(scad_path, stl_path, args.openscad, args.timeout)
        return validate_stl(stl_path, constraints)

    try:
        render_scad_to_stl(scad_path, stl_path, args.openscad, args.timeout)
    except RenderError as exc:
        print(f"error: render failed [{exc.reason}]: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {stl_path}")
    report = validate_stl(stl_path, constraints)
    print(report.human_summary())

    if not report.passed and args.correct:
        adjustable = {f.strip() for f in args.adjustable.split(",") if f.strip()}
        agent = CorrectionAgent(rebuild=rebuild, max_attempts=args.max_corrections)
        try:
            result = agent.correct(params, report, adjustable=adjustable)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for attempt in result.attempts:
            status = "accepted" if attempt.accepted else "rejected"
            print(f"correction #{attempt.index} [{status}]: {attempt.change} — {attempt.detail}")
        correction_path = os.path.join(args.outdir, "correction_log.json")
        with open(correction_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "success": result.success,
                    "detail": result.detail,
                    "model_calls_made": result.model_calls_made,
                    "final_params": result.params.to_dict(),
                    "attempts": [a.__dict__ for a in result.attempts],
                },
                handle,
                indent=2,
            )
        print(f"wrote {correction_path}")
        print(f"correction {'SUCCEEDED' if result.success else 'FAILED'}: {result.detail}")
        report = result.report

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
    print(f"wrote {report_path}")
    return 0 if report.passed else 1


def cmd_probe_dry_run(args) -> int:
    import json as _json

    from reforge.probe import ProbeError, prepare_probe

    try:
        report = prepare_probe(
            args.image,
            args.mask_summary,
            concept=args.concept,
            classification=args.classification,
            model=args.model,
            max_cost_usd=args.max_cost,
            ledger_path=args.ledger,
        )
    except ProbeError as exc:
        print(f"probe dry-run failed: {exc}", file=sys.stderr)
        return 2
    print(_json.dumps(report, indent=2))
    return 0


def cmd_baseten_dry_run(args) -> int:
    import json as _json

    from reforge.probe import ProbeError, prepare_baseten_probe

    try:
        report = prepare_baseten_probe(
            args.image,
            args.mask_summary,
            concept=args.concept,
            classification=args.classification,
            model=args.model,
            max_tokens=args.max_tokens,
            max_cost_usd=args.max_cost,
            ledger_path=args.ledger,
        )
    except ProbeError as exc:
        print(f"baseten dry-run failed: {exc}", file=sys.stderr)
        return 2
    print(_json.dumps(report, indent=2))
    return 0


def cmd_sam_segment(args) -> int:
    from reforge.sam_experiment import run_sam_experiment

    result = run_sam_experiment(
        image_path=args.image,
        concept=args.concept,
        out_path=args.out,
        experiment_id=args.experiment_id,
        ledger_path=args.ledger,
        cache_dir=args.cache_dir,
        max_cost_usd=args.max_cost,
    )
    print(f"[{result.status}] {result.detail}")
    if result.status in ("ok", "cached"):
        print(f"boxes={result.box_count} masks={result.mask_count} overlay={result.overlay_path}"
              + (" (from cache, no new request)" if result.from_cache else ""))
        return 0
    if result.status == "blocked":
        return 3
    return 1


def cmd_budget(args) -> int:
    from reforge.budget import BudgetGuard, BudgetPolicy
    from reforge.usage_ledger import UsageLedger, default_ledger_path

    ledger = UsageLedger(args.ledger or default_ledger_path())
    policy = BudgetPolicy(
        total_limit_usd=args.limit,
        reserve_usd=args.reserve,
        per_experiment_cap_usd=args.per_experiment_cap,
    )
    guard = BudgetGuard(policy, ledger)
    totals = ledger.totals()
    try:
        spent = guard.spent_usd()
        discretionary = guard.discretionary_usd()
    except Exception as exc:  # noqa: BLE001 - guard unavailable is the message
        print(f"budget guard unavailable: {exc}", file=sys.stderr)
        return 1
    print(f"ledger: {ledger.path}")
    print(f"confirmed spend:   ${totals['confirmed_usd']:.2f}")
    print(f"estimated spend:   ${totals['estimated_usd']:.2f}")
    print(f"unknown-cost calls: {totals['unknown_count']}")
    print(f"conservative spend: ${spent:.2f}")
    print(f"hard limit:         ${policy.total_limit_usd:.2f} "
          f"(reserve ${policy.reserve_usd:.2f})")
    print(f"discretionary left: ${discretionary:.2f}")
    print("NOTE: local accounting only; reconcile against the provider dashboard.")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "pipeline":
        return cmd_pipeline(args)
    if args.command == "budget":
        return cmd_budget(args)
    if args.command == "probe-dry-run":
        return cmd_probe_dry_run(args)
    if args.command == "baseten-dry-run":
        return cmd_baseten_dry_run(args)
    if args.command == "sam-segment":
        return cmd_sam_segment(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
