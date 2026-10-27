# Reforge — Manual End-to-End Testing Guide

This manual lets you operate and verify the entire Reforge system without
asking the coding agent. Every command is copy-pasteable. The expected
working directory is the project root unless stated otherwise.

Assumed background: you can use the Ubuntu terminal and VS Code. You do not
need to know project internals; every file path and command name below is
the real one.

> Conventions: blocks labelled `Expected (observed …)` show real output
> captured during development on this machine. Your numbers (timings,
> temp-dir names) will differ slightly; pass/fail verdicts should match.

---

## A. Environment prerequisites

Required:

- Ubuntu Linux with `openscad` 2021.01+ on `PATH`.
- Python 3.10+ (the project virtualenv uses 3.12).
- The project virtualenv at `.venv` (already created in this checkout).
- For live-model tests only: Meta API access via the `MODEL_API_KEY`
  environment variable. **The local pipeline, tests, budget checks, and
  correction demo all run without any credentials.**

1. Enter the project directory:

```bash
cd ~/projects/reforge
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Verify Python and OpenSCAD:

```bash
python --version        # expect: Python 3.12.x
openscad --version      # expect: OpenSCAD version 2021.01 (or newer)
which reforge           # after install: .../reforge/.venv/bin/reforge
```

4. Install project dependencies (editable install; offline-safe for the
   core package):

```bash
pip install -e .
```

5. Check disk space (renders and ledgers are small, but check anyway):

```bash
df -h . | tail -1
```

No environment variables are required for local use. Never paste API keys
into source files, the README, logs, or screenshots; the only consumer of
`MODEL_API_KEY` is the standalone `verify_sam3.py` script, read from the
environment at runtime.

---

## B. Initial setup

1. Install the package (from the project root, venv active):

```bash
pip install -e .
```

This exposes the `reforge` command and makes `python -m unittest`
discover the suite. Re-run it after pulling new changes.

2. Confirm no credentials are in the tree:

```bash
git status --short
grep -rniE 'sk-[A-Za-z0-9]{8,}|bearer [A-Za-z0-9._-]{8,}|"api[_-]?key"[[:space:]]*:[[:space:]]*"[^"]{4,}' \
  --include='*.py' --include='*.md' --include='*.json' --include='*.toml' \
  src tests docs README.md pyproject.toml verify_sam3.py \
  || echo "clean: no embedded secrets"
```

Expected: `clean: no embedded secrets`. (The variable *name*
`MODEL_API_KEY` appears in docs and `verify_sam3.py`; only a stored
*value* would be a leak, and the pattern above targets token shapes.)

3. Local configuration and logs live here:

| Path | Contents |
|---|---|
| `ledger/usage.jsonl` | append-only API usage ledger (created on first paid call) |
| `ledger/cache/` | cached model responses, content-addressed |
| `outputs/` | your renders and reports (git-ignored) |
| `/tmp/reforge-test-*/` | throwaway test dirs (auto-removed) |

4. The application runs fully **without** API credentials. Any command that
   would need a model refuses safely through the budget guard instead of
   crashing (see Section E).

---

## C. Local CAD smoke test

Generate the sample plate (60 × 40 × 5 mm, four Ø6 mm holes):

```bash
cd ~/projects/reforge
source .venv/bin/activate
reforge pipeline --length 60 --width 40 --thickness 5 \
  --hole-diameter 6 --holes "10,10;50,10;10,30;50,30" \
  --outdir outputs/demo
```

Expected output files:

- `outputs/demo/plate.scad` — editable OpenSCAD source.
- `outputs/demo/plate.stl` — exported mesh.
- `outputs/demo/validation_report.json` — machine-readable report.

Expected (observed 2026-10-26) console tail:

```text
wrote outputs/demo/plate.stl
Validation of outputs/demo/plate.stl: PASS (924 facets, 13 checks)
  [ok  ] file_present: STL file exists and is nonempty
  ...
  [ok  ] bbox_axis_X: extent 60.000 mm vs expected 60.000 mm (tol 0.25 mm)
  [ok  ] bbox_axis_Y: extent 40.000 mm vs expected 40.000 mm (tol 0.25 mm)
  [ok  ] bbox_axis_Z: extent 5.000 mm vs expected 5.000 mm (tol 0.25 mm)
  [ok  ] hole_0: centre (10.00, 9.95) vs (10.0, 10.0); diameter 5.95 vs 6.0 ...
wrote outputs/demo/validation_report.json
```

Note the hole reads 5.95 mm, not 6.00: that is real faceting plus the
0.1 mm ray-measurement step, well inside the 0.6 mm hole tolerance. The
outer faces are exact (60.000/40.000/5.000).

Inspect the results:

```bash
cat outputs/demo/plate.scad
python -c "import json; print(json.load(open('outputs/demo/validation_report.json'))['passed'])"
```

Open `outputs/demo/plate.scad` in VS Code or the OpenSCAD GUI
(`File → Open`), press F6 to re-render, and compare with `plate.stl`.
The STL is also re-importable: `import("outputs/demo/plate.stl");` in a
wrapper `.scad` renders cleanly (this is asserted by the test suite).

Validate an existing STL against parameters without re-rendering:

```bash
reforge validate --stl outputs/demo/plate.stl \
  --length 60 --width 40 --thickness 5 \
  --hole-diameter 6 --holes "10,10;50,10;10,30;50,30" \
  --report outputs/demo/recheck.json
echo "exit=$? (0 means PASS)"
```

Success looks like: exit code 0, report `"passed": true`, 13/13 checks ok.

If a step fails: a nonzero exit means render failure (1) or bad parameters
(2); a validation FAIL prints the failing check names — see Section H.

---

## D. Automated test suite

Run the full suite (venv active, project root). It needs no credentials,
no network, and no paid calls — only the local OpenSCAD executable:

```bash
python -m unittest discover -s tests
```

Expected: `Ran 97 tests ... OK` in ~15–20 s (count grows as suites are added).

What each group covers:

| Module | Covers |
|---|---|
| `tests/test_parameters.py` | valid sets; zero/negative/NaN/inf/bool/str dims; bad diameters; holes outside/overlapping; malformed and unknown keys |
| `tests/test_cad_generator.py` | dimension/hole substitution; cutter spans thickness; determinism; injection safety (no untrusted tokens in output) |
| `tests/test_renderer.py` | successful render; missing executable/input; nonzero exit; timeout — all as structured `reason` codes |
| `tests/test_validator.py` | missing/empty/garbage STL; real-mesh PASS; wrong-dims and wrong-diameter FAIL; tolerance actually applied; machine-readable report shape |
| `tests/test_pipeline.py` | end-to-end params → SCAD → STL → validation, plus STL re-import in OpenSCAD |
| `tests/test_budget.py` | permit below limit; block over limit; $25 reserve; unknown-cost conservatism; blocked-request recording; retry caps; cache reuse; guard-unavailable |
| `tests/test_correction.py` | deterministic fix; invalid-proposal rejection; known-field protection; no-improvement stop; attempt/call/budget limits; zero model calls when deterministic fix exists |
| `tests/test_cli.py` | `generate`/`pipeline`/`budget` wiring; correction demo; failing validation exit code |
| `tests/test_sam_experiment.py` | guard blocks before send; no retry; ok/error/cached ledgering; unknown-cost conservatism; cache reuse; no secrets in ledger (all mocked, $0 spent) |

Failure workflow:

```bash
# 1. See which test failed:
python -m unittest discover -s tests 2>&1 | grep -E "^(FAIL|ERROR)"
# 2. Re-run just that test verbosely:
python -m unittest tests.test_validator.TestValidatorOnRealMesh.test_valid_mesh_passes -v
```

Unit tests (parameters, budget, correction logic with fake rebuilds) run in
milliseconds. Integration tests (anything rendering via OpenSCAD, marked
with `@needs_openscad`) take ~1–2 s each and are skipped automatically if
the executable is missing. Mocked tests: only the correction agent's
*model hook* is ever mocked (a local function standing in for a future
diagnose call) — every geometry assertion runs against real OpenSCAD
output. No test makes a network or paid call.

---

## E. Budget guard verification

Nothing here spends credits. All checks use throwaway ledgers.

1. Show the (empty) real ledger state:

```bash
reforge budget --ledger ledger/usage.jsonl
```

Expected on a fresh checkout:

```text
confirmed spend:   $0.00
estimated spend:   $0.00
unknown-cost calls: 0
conservative spend: $0.00
hard limit:         $150.00 (reserve $25.00)
discretionary left: $125.00
```

2. Exercise permit/block paths without spending (unit tests, isolated
   temp ledgers):

```bash
python -m unittest tests.test_budget -v
```

This covers, with zero real charges: a permitted small request; a request
blocked for exceeding the total; a request blocked to protect the $25
reserve; an unknown-cost entry counting conservatively (never $0); an
unbounded request blocked without a manual exception; retry-limit
enforcement; and cache-hit reuse.

3. Manually probe the guard in Python (still no API calls):

```bash
python - <<'EOF'
import sys; sys.path.insert(0, 'src')
from reforge.budget import BudgetGuard, BudgetPolicy
from reforge.usage_ledger import UsageLedger
ledger = UsageLedger('/tmp/guard-probe.jsonl')
guard = BudgetGuard(BudgetPolicy(), ledger)
print('small request:', guard.check('probe-1', 1.0).allowed)      # expect True
print('huge request: ', guard.check('probe-2', 500.0).allowed)     # expect False
print('unbounded:    ', guard.check('probe-3', None).allowed)      # expect False
EOF
```

4. Inspect the ledger (JSON lines; asset hashes and experiment ids only —
   no prompts, images, or secrets):

```bash
cat /tmp/guard-probe.jsonl
```

5. Reconcile with reality: local accounting is conservative, not a billing
   guarantee. Before any paid experiment, compare `reforge budget` totals
   against the provider dashboard (Meta API usage page). If they disagree,
   trust the dashboard and record a correcting entry; never raise the local
   caps to "match" the dashboard.

---

## F. Live model smoke test (single guarded SAM experiment)

Status: the guarded experiment path is implemented and mock-tested
(`src/reforge/sam_experiment.py`, CLI `reforge sam-segment`, legacy
`verify_sam3.py` now delegates to it). **Do not run it until the operator
has supplied a mechanical-part photo and explicitly approved the spend.**
No paid call has been made from this checkout with the guarded path.

The standing proposal for the first (and currently only approved-scope)
experiment is:

- **Question:** does SAM 3.1 segment a top-down mechanical-part photo
  (plate + holes) well enough for contour extraction?
- **Why necessary:** the duck overlay proves the workflow, not mechanical
  reconstruction.
- **Model/operation:** `sam-3.1` image segmentation, one image, concept
  `"flat metal plate with circular holes"`.
- **Expects:** ≥1 mask covering the plate outline; holes visible as gaps.
- **Success:** mask covers the part, background excluded, outer contour
  complete, holes distinguishable as gaps. Failure: fragments, background
  included, or no match.
- **Maximum cost:** one call, operator-declared bound `--max-cost 5.00`
  (the $5 per-experiment cap). Basis and limit of this number: it is a
  **local policy bound, not a provider-side cap** — the provider publishes
  no per-call SAM price to this tool, and the SAM stream returns no charge
  or token usage, so the ledger records post-call cost as `unknown`
  (conservatively counted, never zero). If that basis is not acceptable,
  do not run the call.
- **Decision after:** if the mask is contour-ready, proceed to Muse Spark
  interpretation (Section L criteria); otherwise fix photography first
  (lighting, angle, contrast) before spending more.

### F.1 Supply the image

Copy one top-down, well-lit photo of the flat part here (do not `git add`
personal photos — keep them out of commits):

```bash
cp /path/to/your-part-photo.jpg assets/part.jpg
```

Do not use `assets/duck.jpg` (the duck) as evidence for parts.

### F.2 Verify the budget policy

```bash
cd ~/projects/reforge
source .venv/bin/activate
reforge budget --ledger ledger/usage.jsonl
```

Proceed only if `discretionary left` is at least the declared bound
($5.00) and the dashboard balance agrees with the ledger within the known
gap (see F.5: the pre-guard duck-test spend of roughly $2.26 was never
ledgered — dashboard reads ~$147.74 remaining of $150).

### F.3 Run the experiment (approved spend only)

```bash
export MODEL_API_KEY="..."   # env only; never commit or print it
reforge sam-segment --image assets/part.jpg \
  --concept "flat metal plate with circular holes" \
  --out assets/part_overlay.png \
  --experiment-id sam-mechanical-001 \
  --max-cost 5.00
echo "exit=$? (0 = ok/cached, 1 = error, 3 = blocked)"
```

What this does, in order: hashes the image; reuses the cache when the
identical image+concept already ran (no new request, exit 0, `(from
cache)`); otherwise checks the budget guard **in the same process** and
refuses (exit 3, nothing sent) unless the $5.00 bound fits the cap and the
discretionary budget; then sends **exactly one** streaming request with no
retries; writes the overlay; appends one ledger entry. A second run with
the same image and concept costs nothing — it is served from
`ledger/cache/`.

The equivalent legacy spelling (`python verify_sam3.py assets/part.jpg
"flat metal plate with circular holes" assets/part_overlay.png`) runs the
same guarded path with the same defaults.

### F.4 Inspect the overlay

Open `assets/part_overlay.png` in VS Code or an image viewer. Red must
cover the part, not the background; the outer contour should be complete;
holes should appear as unfilled gaps inside the red region. SAM returns
image-space masks and boxes only — any millimetre value attached later is
an estimate until a known dimension grounds the scale.

### F.5 Inspect the ledger and reconcile with the dashboard

```bash
cat ledger/usage.jsonl
reforge budget --ledger ledger/usage.jsonl
```

The new entry carries `timestamp`, `model: sam-3.1`, `operation:
image_segmentation`, the image `asset_hash`, `experiment_id:
sam-mechanical-001`, box/mask counts, and `cost_status: unknown` (see the
cost-basis note above). To reconcile: match the entry's timestamp and
asset hash against the provider dashboard, then append the actual charge
with this template (fill in the `<...>` values from the dashboard):

```bash
python - <<'EOF'
import sys; sys.path.insert(0, 'src')
from reforge.usage_ledger import UsageLedger
UsageLedger('ledger/usage.jsonl').log(
    provider="meta", model="sam-3.1", operation="reconciliation",
    status="ok", experiment_id="sam-mechanical-001",
    charge_usd=float("<actual USD charge>"), cost_status="confirmed",
    note="dashboard reconciliation for <timestamp>; supersedes the unknown-cost entry")
EOF
```

Apply the same template once for the legacy pre-guard spend (dashboard
shows roughly $2.26 used before the ledger existed) so the guard's
conservative math stays honest.

### F.6 Decide whether to proceed

Proceed to Muse Spark interpretation only if ALL hold: mask covers the
part; background excluded; contour complete enough for extraction; holes
distinguishable; ledger entry present with matching asset hash. Otherwise
stop, keep the overlay as the failure record, and re-shoot before any
further spend. Confirm no retries happened: one experiment id → at most
one `ok`/`error` entry; any re-run is a cache hit, and any genuine retry
is a separate approved run.

### F.7 Baseten dry-run (offline, no approval needed)

```bash
reforge baseten-dry-run --image assets/part.jpg --ledger /tmp/b-ledger.jsonl
```

Expected: `"transmitted": false`, provider `baseten`, endpoint
`https://inference.baseten.co/v1/chat/completions`, model
`meta-models/muse-spark-open-1026`, `has_image: false`, and a guard
verdict. No credentials are read; `/tmp/b-ledger.jsonl` must not be
created. This proves offline assembly only — never provider
compatibility. A live Baseten probe additionally requires: the slug
visible on the account, a per-token price or written acceptance of
unknown cost, a data-URL-or-hosted-image decision, a per-model TOKEN
limit on the key, and explicit per-probe approval. `max_tokens` caps
output length, not dollars.

---

## G. Full end-to-end reconstruction test

This exercises the local reconstruction loop with a deliberate error and a
bounded fix. No API calls, no credentials.

1. Write the known constraints (your "measured" values):

```bash
cat > /tmp/constraints.json <<'EOF'
{"length": 60, "width": 40, "thickness": 5,
 "holes": [{"x": 10, "y": 10, "diameter": 6}]}
EOF
```

2. Run the pipeline with a deliberately wrong inferred hole (14,10 instead
   of 10,10) and correction enabled:

```bash
reforge pipeline --length 60 --width 40 --thickness 5 \
  --holes "14,10,6" --constraints-json /tmp/constraints.json \
  --correct --adjustable holes --outdir outputs/correction-demo
echo "exit=$? (0 means corrected and PASS)"
```

3. Success looks like: initial `hole_0` FAIL (centre ~(14,10) vs (10,10)),
   then `correction #1 [accepted]: set inferred field(s) to
   validation-expected values`, then `correction SUCCEEDED`, exit 0, and
   `correction_log.json` with `"success": true, "model_calls_made": 0`
   and final holes `[{"x": 10.0, "y": 10.0, "diameter": 6.0}]`.

4. Inspect the final artefacts:

```bash
cat outputs/correction-demo/correction_log.json
python -c "import json; print(json.load(open('outputs/correction-demo/validation_report.json'))['passed'])"
```

5. Check the ledger is untouched (correction is deterministic):

```bash
reforge budget --ledger ledger/usage.jsonl | head -4
```

What success means at each step: the wrong candidate is *detected* (not
silently accepted), the fix touches only the adjustable (inferred) field,
the original constraints file is byte-identical afterwards, and the final
report passes against those unweakened constraints.

If a step fails: `correction FAILED` with per-attempt reasons in
`correction_log.json`. A failure naming a user-known field means the
constraints themselves are inconsistent — fix the inputs, not the agent
(see Section H).

---

## H. Negative tests and failure recovery

Each command below is safe (no spending, no damage). Expected behaviour is
taken from the implementation and its tests.

```bash
# Invalid image / missing dimension: parameter validation rejects first.
reforge generate --length 0 --width 40 --thickness 5 --scad /tmp/x.scad
# expect: exit 2, "error: invalid parameters: 'length' must be positive..."

# Impossible hole placement (hole Ø6 at x=19 on a 20 mm plate):
reforge generate --length 20 --width 20 --thickness 3 \
  --holes "19,10,6" --scad /tmp/x.scad
# expect: exit 2, "...extends outside plate length 20.0"

# Unavailable API: nothing in the local path needs it; a blocked model call
# is recorded, never retried silently (unit-covered in tests/test_budget.py).

# Missing OpenSCAD executable:
reforge pipeline --length 60 --width 40 --thickness 5 \
  --holes "10,10,6" --outdir /tmp/no-scad --openscad /nonexistent/openscad
# expect: exit 1, "error: render failed [executable_missing]: ..."

# Rendering failure (invalid SCAD input):
printf 'this is not scad (((' > /tmp/bad.scad
python - <<'EOF'
import sys; sys.path.insert(0, 'src')
from reforge.renderer import render_scad_to_stl, RenderError
try:
    render_scad_to_stl('/tmp/bad.scad', '/tmp/bad.stl')
except RenderError as e:
    print('reason:', e.reason)  # expect: nonzero_exit
EOF

# Invalid STL (validates as failed report, exit 1 — not a crash):
head -c 200 /dev/urandom > /tmp/junk.stl
reforge validate --stl /tmp/junk.stl --length 60 --width 40 --thickness 5
echo "exit=$? (expect 1, report lists stl_parseable FAIL)"

# Budget-blocked request (no spend; blocked entry appended to the ledger):
python -m unittest tests.test_budget.TestBlockedRequests -v

# Unsuccessful correction (constraints contradict user-known values):
cat > /tmp/impossible.json <<'EOF'
{"length": 60, "width": 40, "thickness": 5,
 "holes": [{"x": 10, "y": 10, "diameter": 60}]}
EOF
reforge pipeline --length 60 --width 40 --thickness 5 \
  --holes "10,10,6" --constraints-json /tmp/impossible.json \
  --correct --adjustable holes --outdir /tmp/unsolvable
echo "exit=$? (expect 2: constraints file itself rejected as impossible)"
```

Error-message rule of thumb: exit 2 = your inputs are invalid (fix them);
exit 1 = render or validation failed (read the report); a Python
traceback = a bug — report it with the command and the full output.

---

## I. Troubleshooting

- **Python dependency errors:** re-run `pip install -e .` with the venv
  active. Core needs only `numpy` and `pillow` (both already installed).
- **Virtualenv problems:** the venv lives at `.venv`. Recreate with
  `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`
  (needs the `models` extra only for live-model work).
- **OpenSCAD command failures:** check `openscad --version` and available
  RAM; raise `--timeout` for large parts (default 180 s).
- **Invalid geometry:** read `validation_report.json` — each failed check
  names the property, measured and expected values. Fix inputs; never
  hand-edit the STL.
- **Incorrect scale:** single photos carry no scale. Every millimetre value
  without a user-supplied reference dimension is an estimate — re-run with
  `--constraints-json` holding at least one known measurement.
- **Segmentation failure:** prefer top-down, high-contrast, undistorted
  photos (see Section F). Do not pay for repeat calls on a bad photo.
- **Missing model credentials:** local commands never need them. For live
  tests, `export MODEL_API_KEY=...` (env only) and see Section F.
- **Network errors / rate limits:** local pipeline is unaffected. Live calls
  must be recorded as `status: error` in the ledger with retry discipline
  (`is_retry=True`, max 2).
- **Budget guard blocking a request:** intended behaviour. Read the reason;
  check `reforge budget`; reconcile against the provider dashboard.
- **Disk space shortages:** `df -h .`; clear `outputs/` and `/tmp/*.stl`
  (both regenerable; never delete `assets/` or `ledger/`).

---

## J. Clean final demo procedure

Rehearse with cached/local artefacts (free); verify live integration once,
separately (Section F).

```bash
cd ~/projects/reforge
source .venv/bin/activate
rm -rf outputs/demo outputs/correction-demo   # regenerable outputs only
# 1. Working example + independent validation:
reforge pipeline --length 60 --width 40 --thickness 5 \
  --hole-diameter 6 --holes "10,10;50,10;10,30;50,30" \
  --outdir outputs/demo
# 2. Failed constraint + bounded correction:
cat > /tmp/constraints.json <<'EOF'
{"length": 60, "width": 40, "thickness": 5,
 "holes": [{"x": 10, "y": 10, "diameter": 6}]}
EOF
reforge pipeline --length 60 --width 40 --thickness 5 \
  --holes "14,10,6" --constraints-json /tmp/constraints.json \
  --correct --adjustable holes --outdir outputs/correction-demo
# 3. Spending state: reforge budget --ledger ledger/usage.jsonl
# 4. Open outputs/demo/plate.scad in OpenSCAD (F6) next to
#    outputs/correction-demo/correction_log.json for the narrative.
```

Recording hygiene: never show terminal lines containing `MODEL_API_KEY`
values; the ledger contains no secrets by construction (verify with the
grep in Section B). Say explicitly which artefacts are cached rehearsals
versus the one live-verified call.

---

## K. Expected artifacts and acceptance checklist

Artefacts and where they live:

| Artefact | Path |
|---|---|
| Editable SCAD | `<outdir>/plate.scad` |
| Exported STL | `<outdir>/plate.stl` |
| Validation report (JSON) | `<outdir>/validation_report.json` |
| Correction log (JSON) | `<outdir>/correction_log.json` (only with `--correct`) |
| Usage ledger | `ledger/usage.jsonl` |
| Test suite | `tests/test_*.py` (97 tests, all local, $0 spent) |
| SAM overlay | `--out` PNG from `reforge sam-segment` (red masks over part) |
| SAM cache | `ledger/cache/` content-addressed prior results |
| This manual | `docs/MANUAL_TESTING.md` |

Acceptance checklist:

- [ ] Local CAD generation produces `.scad` that opens in OpenSCAD.
- [ ] STL export succeeds via the local CLI.
- [ ] Independent validation passes on the sample plate (13/13).
- [ ] Live model smoke test performed once, bounded, ledger-recorded
      (or explicitly deferred with the Section F proposal on file).
- [ ] Budget guard permits/blocks correctly without spending (`tests/test_budget.py`).
- [ ] Bounded correction fixes the deliberate error without touching
      known values, within 2 attempts and 0 model calls.
- [ ] Full suite green: `python -m unittest discover -s tests`.
- [ ] Manual followed end-to-end by a human without agent help.
- [ ] No secrets in the tree; ledger reconciled with the dashboard.
- [ ] Demo rehearsed from clean outputs; live vs cached labelled.

---

## L. Known limitations

Stated plainly; none of these are hidden to look impressive:

- Only rectangular plates with circular through-holes perpendicular to the
  face are generated. Brackets, flanges, washers, and spacers are design
  targets, not implemented geometry.
- A single photograph cannot establish scale, thickness, or hidden
  geometry. All such dimensions are estimates until grounded by a known
  measurement (`--constraints-json`).
- Perspective distortion is not corrected; use near-top-down photos.
- Hole verification checks each *expected* hole (presence, centre,
  diameter) plus bbox, closedness, and mesh validity. It does not prove the
  absence of extra unintended holes elsewhere in the part.
- Blind (non-through) holes are not supported or detected.
- Hole measurement resolution is ~0.05–0.1 mm (faceting + ray step);
  tolerances below that will fail by design.
- SAM segmentation runs only through the guarded single-shot path
  (Section F); mask→contour→parameter wiring and Muse Spark interpretation
  are not built yet. The duck overlay proves the SAM workflow only.
- No web UI exists yet (Phase 5, after CLI reliability — which this
  manual establishes).
- The $150 guard is local accounting, not a provider-side cap. Unknown
  costs count conservatively, but only the dashboard is ground truth.

