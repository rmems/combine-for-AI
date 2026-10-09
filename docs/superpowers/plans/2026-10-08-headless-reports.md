# Headless Reports Implementation Plan

> **For agentic workers:** Execute inline with test-driven-development; keep implementation and verification in this checkout.

**Goal:** Regenerate per-track plots and reviewable summaries from archived inputs for GitHub issue #12.

**Architecture:** Add a versioned rendering-input contract for already prepared summaries, not a replacement for the pending evidence/qualification contracts (#45/#49). Separate cohorts by source and complete context; never average in the renderer. Adapt existing GOZ1 imports without modifying legacy readers or claiming qualified execution.

**Tech Stack:** Python 3.14, existing Pydantic and JSON/CSV writers, optional Matplotlib Agg.

**Spec:** https://github.com/rmems/combine-for-AI/issues/12

## Global Constraints

- Preserve failures, unknown values, source hashes, provenance, units, sample/repeat/seed policy and upstream interval definitions.
- Only render prepared uncertainty; do not calculate statistics or universal scores.
- No GPU, dashboard, Linear issue creation or merge. Open PR before 21:30 America/Chicago on October 8, 2026, draft if incomplete.
- Follow guidance read from `docs/agents-md:AGENTS.md`; do not incorporate unrelated PRs #51/#52.

### Task 1: Validated archived inputs and cohort separation

**Files:** `src/combine_for_ai/report_evidence.py`, `tests/test_plot_reports.py`, `tests/fixtures/plot_evidence.sample.json`.

**Interfaces:** `load_report_evidence(paths)` returns source metadata and validated source-isolated cohorts. Rendering records declare track, independent provenance, context, counts and named metrics with units/definitions/scopes and optional SciPy-prepared intervals.

- [x] Write tests that would catch averaging different datasets/seeds, promoting mock values, dropping failed metrics and fabricating missing zeros.
- [x] Run `uv run pytest tests/test_plot_reports.py -q` and verify failure due to absent reporting capability.
- [x] Implement validation and source-isolated cohorts. Reuse GOZ1 import functions for raw and normalized legacy GOZ1 archives; label their incomplete context and fixture/unknown provenance.
- [x] Verify missing/invalid files and unknown schema versions fail with actionable errors; empty archives generate an explicit empty summary.

### Task 2: Headless rendering and reproducible command

**Files:** `src/combine_for_ai/plot_reports.py`, `scripts/plot_reports.py`, `configs/plots.sample.json`, `pyproject.toml`, `uv.lock`, `README.md`, `docs/REPORTING.md`, `.github/workflows/ci.yml`.

**Interfaces:** `generate_reports(paths, output_dir, config_path, plots=True)` writes deterministic JSON/CSV/Markdown and optional PNG plots. CLI accepts repeated `--input`, `--config`, `--output-dir` and `--no-plots`.

- [x] Extend public-interface tests for preserved interval bounds and hashes, optional plotting, real-shaped GOZ1 fixtures, unsupported-only metrics and deterministic regeneration.
- [x] Add optional `plots` extra and versioned config; use existing writers and Matplotlib Agg. No unavailable metric is drawn at zero.
- [x] Keep legacy block views for route agreement and residual drift; draw prepared track metrics only when supported, with explicit provenance/proxy labels and count/interval notes.
- [x] Document one-command reproduction, input contract, compatibility rules and limitations of unimplemented #45/#49 producers.
- [x] Run `uv sync --frozen --extra plots`, `uv run ruff check .`, `uv run pytest -q`, and the existing smoke CLI help. Render and inspect representative fixture, empty/partial and prepared-interval plots.
- [ ] Review diff; commit, push new branch and open PR with `Closes #12`. Verify PR body/base/draft state and report actual delivery state; never merge.
