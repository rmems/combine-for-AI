# Technical debt register

Inventory taken 2026-09-07 against `main` at `6efd00d`.

Every entry below was **reproduced**, not inferred — the evidence column says how.
Ranked by return on investment: confirmed impact and blast radius over the risk
and effort of landing a fix. Line references are against `6efd00d`; entries
already in flight name their PR.

Status legend: **open** · **in review** (PR up) · **done** (merged).

---

## P0 — confirmed live defects with small, safe fixes

### D1 · Report JSON is not valid JSON · *in review — [#32](https://github.com/rmems/combine-for-AI/pull/32)*

`benchmarks/reporting.py:17` · `write_json`

`NaN` and `Infinity` are not JSON, but `json.dump` writes them as bare literals.
`write_json` is the shared writer behind all three report paths — the benchmark
runner, the grok-ozempic importer, and the comparison runner — and metrics reach
it straight from upstream experiment files, where a degenerate block
legitimately yields a `NaN` cosine.

**Evidence** — `import_goz_experiment.py` on a `metrics.json` with a `NaN`
cosine produced a report that a strict parse rejects with `bare NaN`. That
breaks `JSON.parse`, `jq`, Go and Rust consumers alike.

**Fix scope** — sanitize in the one shared writer and pass `allow_nan=False`.
`compare._json_safe` (added by #28) is the same fix applied to one path only;
drop it as a follow-up once both land.

### D2 · `--run-id` path traversal in the importer · *open*

`src/combine_for_ai/goz_import.py:541` · `write_import_reports`

The user-supplied `--run-id` is interpolated straight into the output path with
no validation.

**Evidence** — `--run-id '../escaped'` wrote to
`<output-dir>/json/../escaped.goz-import.json`, i.e. outside the `json/`
subdirectory it was told to write into; more `../` segments escape
`--output-dir` entirely.

`compare.py` gained exactly this guard (`_validate_run_id`) in #28. The sibling
writer never got it — the fix was applied to one caller instead of to the shared
concern.

**Fix scope (small PR)** — lift `_validate_run_id` into
`src/combine_for_ai/run_id.py`, call it from both `write_comparison_reports` and
`write_import_reports`, and reuse the traversal tests already in
`tests/test_compare.py`. Touches `goz_import.py`, so land after #28 or stack on
its branch.

### D3 · Generated artifacts are not ignored · *in review — [#30](https://github.com/rmems/combine-for-AI/pull/30)*

`.gitignore`

Covered only `__pycache__/` and `*.pyc`, while every documented entrypoint
writes into `reports/` by default.

**Evidence** — running the README quickstart left **11 untracked files**, so
`git add -A` silently commits run artifacts and `git status` stops being a
useful signal.

---

## P1 — unsafe defaults and real failure modes

### D4 · The benchmark runner dies outside a git checkout · *open*

`benchmarks/runner.py:65` · `get_git_info`

Two unguarded `subprocess.check_output(["git", ...])` calls with no `try`/`except`
and no `timeout`.

**Evidence** — from a directory that is not a git repository, `get_git_info()`
raises `CalledProcessError: git rev-parse HEAD returned non-zero exit status
128`. Since `build_metadata` calls it before anything else, *every* benchmark run
from an installed wheel, a source tarball, or a Docker layer without `.git`
crashes before doing any work. `benchmarks/telemetry.py` already gets this right
— its subprocess calls pass `timeout=5` and degrade to `None`.

**Fix scope (small PR)** — catch `CalledProcessError`/`OSError`, add a timeout,
fall back to `"unknown"` for commit and branch. Test with a `tmp_path` cwd.

### D5 · Run-id collisions, triplicated · *partly done*

`benchmarks/runner.py:81` · `src/combine_for_ai/goz_import.py:116`
(fixed in `scripts/compare_runs.py` by #28)

Three independent copies of the same timestamp-only id, all with the same
defect: two processes starting in the same millisecond generate the same id,
resolve to the same JSON/CSV/telemetry paths, and silently overwrite each
other — exactly what parallel experiment-matrix jobs do. Each also reads the
clock twice, so the two halves of the stamp can straddle a second boundary.

**Evidence** — proven for `compare_runs.py` with a frozen clock: 64 calls
produced 1 distinct id. Fixed there in #28; the other two copies are unchanged.

**Fix scope (small PR)** — one `run_id.py` helper (pairs naturally with D2),
called from all three sites.

### D6 · Workflow unsafe defaults and a dead trigger · *in review — [#31](https://github.com/rmems/combine-for-AI/pull/31)*

`.github/workflows/*.yml`

No `permissions:` block in any of the four workflows, so every job inherited the
repository-default `GITHUB_TOKEN` scope though none of them write anything. No
`concurrency:` either, so superseded pushes raced to completion across three
workflows. `ci.yml`'s push filter was `[main, issue-*]`, but no branch in this
repo has ever been named `issue-*`. And `manifest-ingestion` / `mini-eval-smoke`
ran on pull requests only, so the checks that gate a merge never re-ran on the
merge result.

---

## P2 — maintainability and coverage

### D7 · The `test-and-lint` job runs no linter · *open*

`.github/workflows/ci.yml:10`

The job is named `test-and-lint`; its steps are `pytest` and
`run_smoke_benchmark.py --help`. There is no ruff, mypy, formatter or coverage
configuration anywhere in the repo — `grep -rn "ruff\|mypy\|black\|coverage"`
over `pyproject.toml`, `.github/workflows/` and `.codacy.yml` returns nothing.
Static analysis is entirely delegated to the Codacy PR bot, which does not gate
local work and is invisible before pushing.

**Fix scope (small PR)** — pin a ruff version in the dev group, add an explicit
`[tool.ruff]` block so CI is reproducible rather than tracking ruff's shifting
defaults, and add a `ruff check` step. Under ruff's default rules the tree has
exactly **one** violation — `E402` in `scripts/run_artifact_smoke.py:11`, the
deliberate `sys.path` shim that `scripts/compare_runs.py:17` already annotates
with `# noqa: E402` — so the gate can go in green with a one-line change. Enable
`I` (import sorting) in a separate mechanical pass; it is ~12 auto-fixable hits
and would otherwise swamp the diff.

### D8 · Test logic lives in workflow heredocs · *open*

`.github/workflows/manifest-ingestion.yml` — roughly 60 of 93 lines

Four inline `python -c` blocks assert manifest validation and dispatch
behaviour: valid manifest accepted, invalid raises `ValidationError`, and all
five `dispatch_artifact` tags. This is test code that cannot run locally, has no
coverage, produces no useful failure output, and silently drifts from
`tests/test_manifest.py`.

**Fix scope (small PR)** — move each assertion into `tests/test_manifest.py`
(parametrized over the dispatch tags), and reduce the workflow to running
pytest plus the loop that loads every `configs/manifests/*.json`, which is the
one part genuinely about the shipped sample files.

### D9 · `telemetry_to_row` maintains its key list twice · *open*

`benchmarks/reporting.py:40`

The `telemetry is None` branch hardcodes 18 keys that must stay in lockstep with
the keys the real branch derives from `asdict(telemetry.system)` plus the
routing fields.

**Evidence** — verified **currently in sync** (18 keys both ways), so this is a
latent hazard, not a live bug: adding one field to `SystemSnapshot` changes the
populated branch and not the `None` branch, and the CSV header (taken from
`rows[0].keys()`) then depends on whether the first row happened to have
telemetry.

**Fix scope** — derive the empty row from the same field list, e.g. build it
from `SystemSnapshot.__dataclass_fields__` with `None` values.

### D10 · Mock accuracy table duplicates the registry · *open*

`benchmarks/models.py:187` · `MockModelAdapter._correct_rate`

Hardcodes a `{profile_name: rate}` dict covering the six profiles in
`default_quantization_registry()`. A seventh profile silently falls back to
`0.8` with no error, quietly changing what the mock reports.

**Fix scope** — move the rate onto `QuantizationProfile` as a field, so the
registry stays the single source of truth.

### D11 · A skipped test with no replacement · *open*

`tests/test_yaml_manifest.py:210` — `pytest.skip("Skipping because pyyaml is
installed and mocking is complex")`

The only skip in the suite. The behaviour it should cover — the error path when
`pyyaml` is missing — is untested, and the reason given is that the test is
awkward to write, not that the case doesn't matter.

**Fix scope** — force the import failure with
`monkeypatch.setitem(sys.modules, "yaml", None)` or a meta-path finder that
raises `ImportError`, and assert the error message.

### D12 · `metrics_to_row` is a dead passthrough · *open*

`benchmarks/reporting.py:35` — `row = asdict(metrics); return row`. Adds a name
and an import edge, nothing else. Inline it or give it a reason to exist.

### D13 · No issue templates or contributing guide · *open — [issue #18](https://github.com/rmems/combine-for-AI/issues/18)*

Already filed and labelled P3; recording it here so the register matches the
tracker.

### D14 · Actions are tag-pinned, not SHA-pinned · *open*

`actions/checkout@v4`, `actions/setup-python@v5`, `astral-sh/setup-uv@v5`,
`actions/upload-artifact@v4`. Mutable tags mean a compromised or retagged
release runs with whatever token scope the job has. Now that D6 has pinned those
scopes to `contents: read`, the exposure is small — worth doing, not urgent.

### D15 · `compare._json_safe` is redundant once D1 lands · *blocked on #28 + #32*

`src/combine_for_ai/compare.py` — a private re-implementation of the sanitizing
walk that D1 puts in the shared writer. Delete once both PRs are in.

---

## Suggested landing order

1. **#28** — hybrid comparison runner (approved, green, all review findings addressed)
2. **#30** — `.gitignore` (config only, zero risk)
3. **#31** — workflow hardening (config only)
4. **#32** — JSON-safe reports (confirmed defect, 6 regression tests)
5. **D2 + D5** — one `run_id.py` PR: traversal guard and collision resistance for
   the importer and the runner *(after #28, which touches `goz_import.py`)*
6. **D4** — runner survives a missing git checkout
7. **D7** — real lint gate
8. **D8** — workflow heredocs become tests
9. **D15**, then the P2 tail
