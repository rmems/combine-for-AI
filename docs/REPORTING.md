# Headless reports from archived evidence

This is the offline rendering capability for [issue #12](https://github.com/rmems/combine-for-AI/issues/12),
not a dashboard, evaluator, profiler or statistics library. It does not run an
engine, qualify a backend, estimate energy or compute a cross-track score.

## Reproduce

From a clean checkout with Python 3.14 and the committed lockfile, one command
installs the optional plot dependencies and regenerates the fixture reports:

```bash
uv run --frozen --extra plots python scripts/plot_reports.py \
  --input tests/fixtures/plot_evidence.sample.json \
  --input tests/fixtures/goz_multiblock_metrics.sample.json \
  --config configs/plots.sample.json --output-dir reports
```

Replace the `--input` paths with your archived JSON files (repeat the argument
for multiple archives). Matplotlib uses the Agg backend; no display server,
GPU, credentials or upstream engine is required. For summaries alone:

```bash
uv run --frozen python scripts/plot_reports.py --input path/to/archive.json --no-plots
```

Outputs:

| Path | Content |
| --- | --- |
| `reports/json/plot-report.json` | Versioned report, source SHA256 hashes, preserved run/link metadata, cohorts, metric definitions, counts and intervals |
| `reports/json/plot-config.json` | Plotting configuration values snapshot; the original input config's byte hash is in the report |
| `reports/csv/plot-report.csv` | One row per upstream observation, including source/config hashes, context, raw references and availability (omitted for empty input) |
| `reports/markdown/plot-report.md` | Definitions, provenance, sample/repeat/seed policy, status counts, upstream denominators, interval definitions and linked plots |
| `reports/plots/<cohort>-<metric-index>.png` | Available metrics only, with units, provenance, status counts, uncertainty policy and source hash. Per-observation counts, intervals and notes stay in the Markdown and JSON; the figure footnote is bounded so large cohorts still render. |

The command overwrites its fixed summary files; use a separate output directory
for each campaign you want to retain. Older PNGs are not deleted, but **only
images linked by the current JSON/Markdown report belong to the current report**.
An empty regeneration removes the generator's stale CSV. Identical inputs,
paths, config and locked dependency environment produce identical output bytes;
there are no generated timestamps or random filenames. Different Matplotlib/font
versions or platforms can change PNG bytes. Source paths are part of cohort IDs.

## Prepared rendering input (`combine.plot-evidence.v1`)

[The multi-track fixture](../tests/fixtures/plot_evidence.sample.json) is a complete,
executable example. Its values/intervals are shape-only fixtures, **not real
measurements or statistically qualified results**. The Pydantic models in
`src/combine_for_ai/report_evidence.py` define the validated rendering contract.

Root fields are `schema_version`, optional `run` (engine-owned run IDs, URLs and
metadata), and `records` (an empty array is valid). Each record supplies:

| Field | Contract |
| --- | --- |
| `workload_track` | `language`, `agent`, `temporal`, or `compression-analysis`; not an architecture or universal score |
| `execution_kind` | Independently: `mock`, `metadata-only` (header-only inspection), `import`, `execution`, or `unknown` |
| `input_provenance` | Independently: `fixture`, `actual`, or `unknown`; imports may contain fixtures |
| `metric_provenance` | Independently: `measured`, `imported-measurement`, `calculated-proxy`, or `unknown` |
| `series` | Upstream observation/arm label; include model/checkpoint identity here or in raw references; no averages are generated |
| `context` | Frozen comparison settings; absent/incomplete context isolates the record |
| `sample_count`, `repeat_count`, `seed_policy` | Upstream counts and experimental repeat/seed policy, not inferred from array lengths or token counts; unknown counts are null |
| `block_index` | Optional nonnegative block index; enables per-arm block views |
| `source_artifacts` | Optional engine-owned raw references/hashes/links, preserved unchanged; referenced files are not followed or rehashed |
| `metrics` | Named, already prepared metric summaries; may be empty (availability unknown) |

Metrics require `name`, `unit`, `definition`, `scope` and `status`. Any upstream
metric can be shown: accuracy/error, latency, throughput, memory, agent
success/outcomes/budget use, temporal operations and spikes. Definitions must
state measurement boundaries and denominator rules, including treatment of
failures, warmup, batching and process/device scope where relevant. Do not label
algorithmic operation counts or estimated energy as measured device energy;
put proxies in separate records with `metric_provenance: calculated-proxy`.
The renderer displays these declarations; it cannot certify their truth.

- `status: ok` means the metric is **available**, not that an experiment passed
  qualification. It requires a finite numeric `value`, including a genuine zero.
- `failed`, `skipped`, `interrupted`, `unsupported`, `missing`, or `unknown`
  requires `value: null` (or omitted); unavailable metrics cannot have intervals.
  They stay in tables/status counts and never become zero-valued plot points.
- Optional `counts` is the upstream count map using the same statuses. Omitted
  counts remain unavailable, not zero. Observation-row counts are shown
  separately and explicitly are **not sample denominators**. No failure filters,
  denominator adjustments or statistical calculations happen in rendering.
- Optional `notes` preserves upstream caveats or qualification/gate failures.
- Optional `interval` requires `low`, `high`, `confidence_level` (0–1 exclusive),
  `method`, `experimental_unit`, `library: scipy`, `library_version`, and
  `definition`. Bounds must be finite and ordered. Endpoints are plotted directly
  even if the interval does not contain the estimate. Missing uncertainty is
  labelled unavailable; no standard error or confidence interval is invented.

The record provenance validation rejects `measured` for mock, metadata-only or
import runs, and rejects `imported-measurement` outside import runs. Fixtures can
contain actual measured execution of fixture workloads, but **fixture provenance
always remains visible** and does not qualify real-data performance claims.
Do not mix metrics with different provenance in one record.

## Cohorts and comparison safety

Sources are always separate. Within a prepared archive, records share a cohort
only when the **entire** context, track, execution/input/metric provenance,
sample/repeat counts and seed policy match. For grouping, context must include
nonempty, known values for all of:

`task`, `scorer`, `dataset`, `split`, `sample_ids`, `engine`, `config`, `runtime`,
`harness`, `environment`, `budgets`, `preprocessing`, and `seeds`.

Use versioned task/scorer/dataset/engine identities and frozen configuration,
runtime/harness/environment settings. Record sample IDs and seeds explicitly;
repeated samples are not independent seeds. Nested null/unknown/empty values
anywhere in `context`, including fields outside the required set, make that
record incomplete. A string such as `none` may explicitly declare that a
setting (e.g. preprocessing) does not apply. Missing context is labelled
`isolated: incomplete context`, never inferred from matching metric names.

Even in a compatible cohort, different metric units/definitions/scopes get
separate plots. Each series remains an individual upstream prepared summary;
the renderer **never pools, averages, ranks or computes deltas**. Duplicate
series/metric/block observations are rejected even when incomplete context
would otherwise isolate them. Prepare repeated observations and their
uncertainty upstream instead of treating repeats as independent results.

## Legacy GOZ1 compatibility and limits

Raw grok-ozempic multiblock (`chain.per_block`) and route-preservation
(`pilot` + `summary`) JSON, plus normalized `*.goz-import.json` reports with a
supported GOZ1 schema, use the existing importer. Existing readers/reports are
unchanged. Retained views include top-1/top-2 route agreement and residual drift
vs block, block-output cosine and expert-load JS divergence. Missing expected
`expert_only`/`fp16_control` arms or declared blocks are visible as missing.
Malformed rows and unknown schema versions fail closed, not silently filtered.
A present legacy metric that is non-numeric, boolean, or non-finite is rejected;
null and omitted metrics stay missing. An empty normalized result object is
rejected rather than shown as an unknown series. Imported `source_path` values
are provenance labels in cohort context and are not opened or resolved.

Legacy data is labelled `import / fixture-or-unknown / unknown` and
`legacy observations: unqualified`. Explicit fixture markers are honored;
filenames, certification booleans, speed fields or apparent real-world values
never imply measured execution. Token counts are preserved in context but **not
converted to sample counts**. The route-preservation report's threshold `fail`
statuses stay in notes alongside their available observations; a gate failure is
not an execution failure. Legacy imports that discarded gate metadata cannot
reconstruct it: the report labels that status unknown and retains the archive hash.

Old benchmark/matrix speed and VRAM presets lack sufficient provenance and are
not automatically plotted as performance. They get an actionable error asking
for prepared summaries; their legacy writers/readers remain usable unchanged.
CSV-only inputs are not accepted: use the richer paired archived JSON to avoid
losing context, provenance or uncertainty definitions.

## Relationship to qualification (#45/#49)

The shared evidence and statistical qualification producers in #45/#49 are still
separate work. This small versioned contract is a **rendering boundary**, not a
new evaluator or proof of qualification. Future producers can export validated
engine-owned summaries, with uncertainty prepared under #49 using SciPy, into
this format. The renderer only consumes and displays that uncertainty; SciPy is
not a runtime rendering dependency. Importing an archive or passing fixture
tests does not independently verify upstream score parity, licenses, hardware
measurement boundaries, raw referenced hashes or live-agent capability.

## Validation

```bash
uv sync --frozen                         # core-only environment
uv run ruff check .
uv run pytest -q
uv sync --frozen --extra plots           # optional headless rendering coverage
uv run pytest tests/test_plot_reports.py -q
```

CI runs both core and plots environments. Public-interface tests cover missing,
empty, malformed, partial, incompatible and real-shaped multi-track/GOZ1 fixture
archives, preserving uncertainty/failures/hashes and deterministic PNG regeneration.
