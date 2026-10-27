# Reforge

Turn a photograph of a simple flat mechanical part into editable parametric
CAD — with every AI-generated dimension checked locally before it becomes
geometry.

## Problem

Going from a reference photo of a bracket, plate, or flange to usable CAD
today means eyeballing dimensions, redrawing from scratch, and hoping the
holes land where the part says they are. Each transcription step invents
precision the photo never contained. Reforge connects visual evidence to
constrained, locally validated geometry instead: models propose,
deterministic code disposes — parameters render to CAD only after passing
independent mesh validation, and anything uncertain stays explicitly
unconfirmed.

## Target user

A solo maker, hobby machinist, or repair technician who needs a simple flat
replacement part redrawn from a photo. (No user research or customer
validation has been done; this is a hackathon prototype, not a product.)

## What is novel

The intended evidence-to-request-to-validated-geometry loop: SAM
segmentation supplies image-space evidence, a Spark request asks for a
strict uncertainty-aware geometry proposal, a deterministic bridge converts
only confirmed, scale-grounded values into parameters, and bounded
correction repairs failures without ever weakening user measurements —
with every paid call gated by a budget guard and recorded in a local
ledger. Implemented today: the full local half (parameters → CAD →
validation → correction → accounting) plus offline Spark request
preparation and a disabled-by-default transport. Not yet demonstrated:
live Spark inference flowing into CAD. See Limitations.

## Current capabilities (all locally testable, $0 spent)

- Parametric plates with through-holes → OpenSCAD source, STL export,
  independent mesh validation (bbox, hole presence/centre/diameter,
  closedness), bounded deterministic correction.
- One guarded SAM segmentation experiment (cached result + overlay).
- Offline Spark request preparation: `probe-dry-run` builds the Responses
  wire body and reports the guard verdict without transmitting.
- 316-test suite, fully offline; both HTTP transports (Meta Responses,
  Baseten Chat Completions) have only ever run against injected fakes and
  are unreachable by default. `baseten-dry-run` assembles the Baseten
  body offline and reports `transmitted: false`.

## Architecture

```text
photo → mask evidence → SparkRequest → wire body → [transport: disabled]
        → parsed proposal → schema + CAD pre-scan → bridge → PlateParams
        → trusted SCAD template → OpenSCAD → STL → validator → correction
```

Modules: `parameters`, `cad_generator`, `renderer`, `validator`,
`correction_agent`, `geometry` (schema), `mask_evidence`, `proposal_bridge`,
`spark_request` + `spark_schema`, `http_transport` (default-off),
`sam_experiment`, `budget` + `usage_ledger`, `probe`, `cli`. Paid calls go
through `BudgetGuard` ($150 ceiling, $25 reserve) into an append-only
JSONL ledger; that tripwire is local policy, not a provider billing cap.

## Quickstart

Prerequisites: Ubuntu, Python 3.10+, OpenSCAD on `PATH`.

```bash
cd ~/projects/reforge
source .venv/bin/activate   # project virtualenv (Python 3.12 here)
pip install -e .            # editable install; core needs numpy + pillow
reforge pipeline --length 60 --width 40 --thickness 5 \
  --hole-diameter 6 --holes "10,10;50,10;10,30;50,30" \
  --outdir outputs/demo     # verified 2026-10-27: PASS, 13/13 checks
```

Offline Spark dry-run (proves wiring, sends nothing):

```bash
reforge probe-dry-run --image assets/part.jpg --ledger /tmp/demo-ledger.jsonl
# prints "transmitted": false plus evidence, guard verdict, and spend state
```

Never run `sam-segment` without explicit spend approval — it is the one
command that can place a paid call.

## Validation and tests

```bash
python -m unittest discover -s tests   # verified 2026-10-27: Ran 316, OK
```

The suite proves local behavior only: geometry math, validation, guard
accounting, mocked transports, and the disabled-by-default gate. It does
not prove live-endpoint compatibility — no test touches the network, and
no test has ever run against the real provider.

## Baseten Muse Spark Open preparation (offline, unverified)

A separate, isolated Chat Completions path exists for the hackathon's
`meta-models/muse-spark-open-1026` slug (operator-supplied): base
`https://inference.baseten.co/v1`, endpoint `/chat/completions`, with the key read from `BASETEN_API_KEY`
only (never `MODEL_API_KEY`), stdlib HTTP, single-shot, no retries,
guard-first, a `max_tokens` output cap, and the
full local marker-scan → schema → CAD validation pipeline on any reply.

```bash
reforge baseten-dry-run --image <photo.jpg> --ledger /tmp/ledger.jsonl
# prints "transmitted": false plus endpoint, guard verdict, and open questions
```

Not verified: the slug on the operator account, per-token price and
credit applicability, and base64 `data:` image delivery (Baseten documents
server-fetched HTTPS URLs, so dry-run sends no image and unapproved image
modes fail closed). Mocked tests prove offline behavior only — never
provider compatibility. `max_tokens` limits output length, not dollars:
input/image tokens also bill, so no dollar ceiling is claimed. A live
probe needs the operator's per-item checklist sign-off; approving the
code release does not approve any live call.

## Usage accounting and safety

`reforge budget` shows confirmed / estimated / unknown-cost spend plus the
conservative total the guard enforces. Current local state: $0 confirmed,
one unknown-cost SAM call ($5 conservative allowance, $120 discretionary).
The Duck (~$2.26) and SAM actual charges are unreconciled against the
provider dashboard — the ledger understates reality until the operator
appends dashboard-backed entries. Local caps never limit provider billing.

## Limitations and roadmap

- Phase 1 (local CAD + validation): COMPLETE. Phase 4 (bounded
  correction): COMPLETE per tests. Phase 2: experiment done, billing
  unreconciled. Phase 3: PARTIAL — no live Spark output has passed
  validation into CAD. Phase 5: NOT IMPLEMENTED (no UI).
- Single photos carry no scale/thickness; the SAM mask includes holes
  (verified locally: 0 interior gaps), so holes need independent evidence.
- Streamlit, live probe, dashboard reconciliation, and demo video are
  future work, each gated on operator approval where spend is involved.

## Hackathon notes

Built for the Meta Global AI Developer Hackathon; Meta tooling used: SAM
3.1 segmentation (live, once) and Muse Spark preparation (offline), plus
an offline Baseten Chat Completions adapter for the hackathon's
Muse Spark Open slug (never run live).
Deadline and submission-page details are operator-confirmed only. No
collaborators; no external validation claimed.

## License and asset rights

Code: MIT — see [LICENSE](LICENSE), copyright Yahia A. Benyahia.
`assets/` photos (`part.jpg`, `part2.jpeg`, `part3.jpeg`, `duck.jpg`,
overlays) are operator-supplied, stated to be Gemini-generated and free
to use, and are **excluded from the first public release**; they must not
be treated as cleared for redistribution without the operator's own
confirmation. Dependencies
(numpy, pillow, openai, meta-sam-parser) are permissively licensed;
no third-party code is vendored.

## Full manual

[docs/MANUAL_TESTING.md](docs/MANUAL_TESTING.md) — environment, smoke
tests, suite, budget verification, live-model procedure, end-to-end
reconstruction, negatives, troubleshooting, demo, checklist, limitations.
