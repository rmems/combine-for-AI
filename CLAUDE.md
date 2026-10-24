# CLAUDE.md

Claude-specific notes only. There is no `AGENTS.md` in this repo; project overview and ecosystem boundaries
are in `README.md` and `docs/ARCHITECTURE.md`, so they are not repeated here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Python 3.14 with uv (Linux, CPU only).
Python is pinned by `.python-version` (3.14); dependencies are locked in `uv.lock`, so use `--frozen`.

| Command | Status |
|---|---|
| `uv sync --frozen` | verified |
| `uv run ruff check .` | verified |
| `uv run pytest -q` | verified (513 passed, 1 skipped) |
| `uv run python scripts/run_smoke_benchmark.py --help` | verified |
| `uv run python scripts/benchmark.py --config configs/benchmark.sample.json` | verified |
| `uv run python scripts/run_matrix.py --config configs/matrix/corinth_canal.sample.json` | verified (22 cells, 0 failed) |
| `./scripts/run_smoke_benchmark.sh configs/manifests/safetensors_hf.sample.json` | verified |
| `uv run python scripts/run_smoke_benchmark.py --output-dir reports/smoke --formats json,csv` | not run here (same smoke path as the CI jobs `benchmark-smoke` / `mini-eval-smoke`) |
| `./scripts/run_smoke_benchmark.sh configs/manifests/goz1.sample.json` | not run here (fails closed by design until `generated_artifacts[].path` is a real `.goz1` pack) |

CI (`.github/workflows/ci.yml`) runs `uv sync --frozen`, `ruff check .`, `pytest -q`, and the smoke `--help`.
Three more workflows (`benchmark-smoke`, `manifest-ingestion`, `mini-eval-smoke`) run the smoke and manifest checks.
Run `ruff check .` and `pytest -q` before committing; CI fails on either.

## Architecture map

- `src/combine_for_ai/`: installable package. `manifest.py` (magere-brug handoff ingestion and dispatch), `goz_import.py`
  (grok-ozempic experiment JSON to report rows), `compare.py` (baseline vs treatment arms), `matrix*.py` (models x quant x datasets
  campaigns), `environment*.py` (sanitized environment fingerprints), `jsonio.py`.
- `benchmarks/`: runner, metrics, scorers, reporting, telemetry, corinth-canal ingest, and the dataset loaders and cache
  (`arc_easy/`, `gsm8k/`, `hellaswag/`, `lambada/`, `piqa/`, `wikitext/`, `dataset_*.py`).
- `scripts/`: CLI entrypoints (`benchmark.py`, `run_matrix.py`, `compare_runs.py`, `import_goz_experiment.py`, smoke runners).
- `configs/`: sample benchmark, manifest, matrix, and offline dataset fixtures. `tests/fixtures/`: pinned goldens and sample inputs.
- `docs/ARCHITECTURE.md` is the design reference. `docs/TECH_DEBT.md` is the known-defect register; check it before touching
  `benchmarks/reporting.py` or `goz_import.py`.

## Invariants and non-goals (from README and `docs/ARCHITECTURE.md`)

- This is an evaluation and reporting harness. It does not pack GOZ1 files, run quantization or residual experiments, or own CUDA kernels.
  GOZ1 format truth lives in `rmems/grok-ozempic`; here only the header (magic, version, tensor count) is sniffed.
- GOZ1 smoke fails closed: it must not silently fall back to HF or another format.
- Dataset loaders work offline from `configs/datasets/*.sample.jsonl`; do not add network fetches to tests or smoke paths.
- Golden scorers are deterministic and pinned in `tests/fixtures/golden_scorers.json`; failures must name dataset, example id, expected, observed.
- MoE/SNN report fields are nullable; keep them nullable when extending the report schema.
- Report JSON should be strict JSON (no bare `NaN`/`Infinity`); see `docs/TECH_DEBT.md` D1.
- Ruff rules are selected explicitly in `pyproject.toml`; do not widen them inside an unrelated change.
- Source files here do not use SPDX headers; do not add them.

## Gotchas

- `requires-python >=3.14`: system Python 3.13 will not do; let `uv` provision 3.14.
- Generated output under `reports/` is gitignored (only `.gitkeep` files are tracked), so sample runs leave the tree clean.
- Never run `uv publish` or any other publish command.
- Edit only what the task needs.
