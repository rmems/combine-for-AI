# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

`combine-for-AI` is a neutral benchmark harness for hybrid MoE/SNN quantization experiments
(see `README.md`). It owns evaluation scripts and dataset loaders, artifact manifest ingestion
(magere-brug handoff JSON/YAML), metric collection, CSV/JSON/Markdown reports, and the GOZ1/SAAQ
evaluation path for `rmems/grok-ozempic` packs. It does **not** own quantization kernels or GOZ1
packing (`grok-ozempic`, `myelin-accelerator`), precision-tier inventory (`xai-dissect`), or SAAQ
calibration loops (`corinth-canal`, `magere-brug`).

Distribution / import names: `combine-for-AI` (slug), `combine-for-ai` (PyPI), `combine_for_ai` (import).

## Layout

| Path | Contents |
|------|----------|
| `src/combine_for_ai/` | Library package |
| `scripts/` | CLI entry points (`benchmark.py`, `run_smoke_benchmark.py`, ...) |
| `benchmarks/` | Benchmark definitions |
| `configs/` | Sample benchmark, dataset, manifest and matrix configs |
| `tests/` (+ `tests/fixtures/`) | pytest suite |
| `reports/`, `docs/` | Generated/sample reports and documentation |

## Toolchain

- Python **3.14** (`.python-version`; `requires-python >=3.14`).
- `uv` with the committed `uv.lock`. CI uses `astral-sh/setup-uv` with uv `0.6.x`.
- No GPU or CUDA requirement.

## Commands (from `.github/workflows/ci.yml`)

```bash
uv sync --frozen
uv run ruff check .
uv run pytest -q
uv run python scripts/run_smoke_benchmark.py --help
```

Other workflows: `benchmark-smoke.yml` and `mini-eval-smoke.yml` run
`uv run python scripts/run_smoke_benchmark.py ...` and `uv run pytest tests/test_smoke.py -q`;
`manifest-ingestion.yml` runs `uv run pytest tests/test_manifest.py -q`.

## Conventions visible in the repo

- Lint with ruff using the rule set selected explicitly in `pyproject.toml` (`[tool.ruff.lint]`,
  `target-version = "py314"`). Import sorting (`I`) is intentionally not enabled yet.
- pytest runs with `pythonpath = [".", "src"]` (`[tool.pytest.ini_options]`).
- Use `uv sync --frozen`; don't regenerate `uv.lock` unless the change is about dependencies.
- Commit subjects mostly follow Conventional Commits (`feat(matrix): ...`, `fix: ...`, `ci: ...`)
  with the PR number appended.
