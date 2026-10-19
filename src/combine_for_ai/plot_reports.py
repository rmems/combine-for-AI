"""Regenerate headless reports from archived observations, never statistics."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from benchmarks.reporting import write_csv
from combine_for_ai.jsonio import ensure_dir, write_json
from combine_for_ai.report_evidence import RenderingModel, load_report_evidence, read_json


class PlotConfig(RenderingModel):
    schema_version: Literal["combine.plot-config.v1"]
    width_inches: float = Field(default=9, ge=6, le=20)
    height_inches: float = Field(default=5, ge=4, le=20)
    dpi: int = Field(default=120, ge=72, le=300)


@dataclass(frozen=True)
class _Output:
    directory: Path
    config: PlotConfig
    config_hash: str
    plots: bool


_PLOT_PATH = re.compile(r"^plots/[0-9a-f]{16}-\d{2,}\.png$")
_TABLE_HEADER = (
    "| Series | Block | Value | Status | Upstream counts "
    "(denominator policy in definition) | Prepared interval | Notes | Raw artifacts |"
)
_TABLE_DIVIDER = "| --- | --- | --- | --- | --- | --- | --- | --- |"
_MARKDOWN_ESCAPES = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ("|", "&#124;"),
    ("!", "&#33;"),
    ("[", "&#91;"),
    ("]", "&#93;"),
    ("(", "&#40;"),
    (")", "&#41;"),
    ("`", "&#96;"),
    ("\n", " "),
    ("\r", " "),
)


def _interval_text(interval: dict[str, Any] | None) -> str:
    if interval is None:
        return "unavailable (no prepared uncertainty)"
    level = interval["confidence_level"] * 100
    return (
        f"[{interval['low']:g}, {interval['high']:g}]; "
        f"{level:g}% {interval['method']}; "
        f"unit={interval['experimental_unit']}; "
        f"{interval['library']} {interval['library_version']}; {interval['definition']}"
    )


def _counts_text(counts: dict[str, int]) -> str:
    if not counts:
        return "unavailable"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _count_or_unknown(value: int | None) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _caption(cohort: dict[str, Any], metric: dict[str, Any]) -> str:
    statuses = Counter(obs["status"] for obs in metric["observations"])
    samples = _count_or_unknown(cohort["sample_count"])
    repeats = _count_or_unknown(cohort["repeat_count"])
    observed = _counts_text(dict(statuses))
    return (
        f"n={samples}; repeats={repeats}; seed policy: {cohort['seed_policy']}\n"
        f"Observation statuses: {observed} (not sample denominators)"
    )


def _cell(value: Any) -> str:
    text = "unavailable" if value is None else str(value)
    for old, new in _MARKDOWN_ESCAPES:
        text = text.replace(old, new)
    return text


def _markdown(payload: dict[str, Any]) -> str:
    lines = _intro(payload)
    lines.extend(_source_lines(payload["sources"]))
    if not payload["cohorts"]:
        lines.extend(["", "No observations; no plots generated.", ""])
    for cohort in payload["cohorts"]:
        lines.extend(_cohort_lines(cohort))
    return "\n".join(lines) + "\n"


def _intro(payload: dict[str, Any]) -> list[str]:
    version = payload["plot_config"]["schema_version"]
    digest = payload["plot_config_sha256"]
    return [
        "# Archived per-track reports",
        "",
        "No averaging or uncertainty estimation is performed here. "
        "Availability statuses are not pass/fail qualification gates.",
        "Fixtures, mock/header-only execution, imports, measured execution and "
        "calculated-proxy metrics remain distinct. Operation/energy proxies are "
        "not device energy measurements.",
        "",
        f"Plot configuration: `{version}`; SHA256 `{digest}`.",
        "",
        "## Sources",
        "",
    ]


def _source_lines(sources: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for source in sources:
        lines.extend(_one_source(source))
    return lines


def _one_source(source: dict[str, Any]) -> list[str]:
    run = _cell(json.dumps(source["run"], sort_keys=True))
    provenance = _cell(json.dumps(source["provenance"], sort_keys=True))
    return [
        f"- {_cell(source['path'])}: SHA256 `{source['sha256']}`",
        f"  Run/link metadata: {run}",
        f"  Source provenance: {provenance}",
    ]


def _cohort_lines(cohort: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## {_cell(cohort['workload_track'])} — cohort {cohort['id']}",
        "",
        _evidence_line(cohort),
        (
            f"Comparison policy: {_cell(cohort['comparison_policy'])}. "
            f"Source: {_cell(cohort['source'])}."
        ),
        f"Context: {_cell(json.dumps(cohort['context'], sort_keys=True))}",
        "",
    ]
    if not cohort["metrics"]:
        lines.append("No metrics supplied; availability and failure counts unknown.")
    for metric in cohort["metrics"]:
        lines.extend(_metric_lines(cohort, metric))
    return lines


def _evidence_line(cohort: dict[str, Any]) -> str:
    kind = _cell(cohort["execution_kind"])
    inputs = _cell(cohort["input_provenance"])
    metric = _cell(cohort["metric_provenance"])
    return f"Evidence: {kind} / {inputs} / {metric}."


def _metric_lines(cohort: dict[str, Any], metric: dict[str, Any]) -> list[str]:
    lines = [
        f"### {_cell(metric['name'])} [{_cell(metric['unit'])}]",
        "",
        f"Definition: {_cell(metric['definition'])}. Scope: {_cell(metric['scope'])}.",
        _cell(_caption(cohort, metric)),
        "",
        _TABLE_HEADER,
        _TABLE_DIVIDER,
    ]
    for obs in metric["observations"]:
        lines.append(_observation_row(obs))
    lines.extend(_plot_link(metric))
    return lines


def _observation_row(obs: dict[str, Any]) -> str:
    cells = [
        obs["series"],
        obs["block_index"],
        obs["value"],
        obs["status"],
        _counts_text(obs["counts"]),
        _interval_text(obs["interval"]),
        obs["notes"],
        json.dumps(obs["source_artifacts"], sort_keys=True),
    ]
    return "| " + " | ".join(_cell(value) for value in cells) + " |"


def _plot_link(metric: dict[str, Any]) -> list[str]:
    plot = metric.get("plot")
    if not plot:
        return ["", "Plot not generated (disabled or all values unavailable).", ""]
    if _PLOT_PATH.fullmatch(str(plot)) is None:
        raise ValueError(f"refusing unexpected plot path: {plot}")
    return ["", f"![{_cell(metric['name'])}](../{plot})", ""]


def generate_reports(
    paths: list[Path],
    output_dir: Path,
    config_path: Path,
    *,
    plots: bool = True,
) -> dict[str, Path]:
    """Validate all inputs before writing reproducible reports under output_dir."""
    config, config_hash, raw_config = _load_plot_config(config_path)
    _require_plots_extra(plots)
    sources, cohorts = load_report_evidence(paths)
    output = _Output(Path(output_dir), config, config_hash, plots)
    rows = _observation_rows(cohorts, sources, output)
    payload = _payload(config, config_hash, sources, cohorts)
    return _write_summaries(output.directory, payload, raw_config, rows)


def _load_plot_config(config_path: Path) -> tuple[PlotConfig, str, dict[str, Any]]:
    raw_config, config_hash = read_json(Path(config_path))
    try:
        config = PlotConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"invalid plot config {config_path}: {exc}") from exc
    return config, config_hash, raw_config


def _require_plots_extra(plots: bool) -> None:
    if not plots:
        return
    # Optional dependency: summary-only runs must not import Matplotlib.
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "Plotting requires optional dependencies: "
            "uv sync --extra plots; or use --no-plots"
        ) from exc


def _payload(
    config: PlotConfig,
    config_hash: str,
    sources: list[dict[str, Any]],
    cohorts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "combine.plot-report.v1",
        "plot_config": config.model_dump(),
        "plot_config_sha256": config_hash,
        "sources": sources,
        "cohorts": cohorts,
    }


def _observation_rows(
    cohorts: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    output: _Output,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        source_hash = _source_hash(sources, cohort)
        for index, metric in enumerate(cohort["metrics"]):
            _maybe_plot(cohort, metric, index, source_hash, output)
            rows.extend(_csv_rows(cohort, metric, source_hash, output.config_hash))
    return rows


def _source_hash(sources: list[dict[str, Any]], cohort: dict[str, Any]) -> str:
    for source in sources:
        if source["path"] == cohort["source"]:
            return str(source["sha256"])
    raise ValueError(f"source not found for cohort: {cohort['source']}")


def _maybe_plot(
    cohort: dict[str, Any],
    metric: dict[str, Any],
    index: int,
    source_hash: str,
    output: _Output,
) -> None:
    if not output.plots or not _has_value(metric):
        return
    # Optional dependency: plot_render imports Matplotlib.
    from combine_for_ai.plot_render import render_plot

    relative = f"plots/{cohort['id']}-{index:02d}.png"
    render_plot(
        output.directory / relative,
        output.config,
        {
            "cohort": cohort,
            "metric": metric,
            "source_hash": source_hash,
            "caption": _caption(cohort, metric),
        },
    )
    metric["plot"] = relative


def _has_value(metric: dict[str, Any]) -> bool:
    return any(obs["value"] is not None for obs in metric["observations"])


def _csv_rows(
    cohort: dict[str, Any],
    metric: dict[str, Any],
    source_hash: str,
    config_hash: str,
) -> list[dict[str, Any]]:
    return [_csv_row(cohort, metric, obs, source_hash, config_hash) for obs in metric["observations"]]


def _csv_row(
    cohort: dict[str, Any],
    metric: dict[str, Any],
    obs: dict[str, Any],
    source_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "cohort": cohort["id"],
        "track": cohort["workload_track"],
        "execution_kind": cohort["execution_kind"],
        "input_provenance": cohort["input_provenance"],
        "metric_provenance": cohort["metric_provenance"],
        "source": cohort["source"],
        "source_sha256": source_hash,
        "plot_config_sha256": config_hash,
        "context": json.dumps(cohort["context"], sort_keys=True),
        "series": obs["series"],
        "block": obs["block_index"],
        "metric": metric["name"],
        "unit": metric["unit"],
        "definition": metric["definition"],
        "scope": metric["scope"],
        "value": obs["value"],
        "status": obs["status"],
        "sample_count": cohort["sample_count"],
        "repeat_count": cohort["repeat_count"],
        "seed_policy": cohort["seed_policy"],
        "counts": json.dumps(obs["counts"], sort_keys=True),
        "interval": json.dumps(obs["interval"], sort_keys=True),
        "notes": obs["notes"],
        "source_artifacts": json.dumps(obs["source_artifacts"], sort_keys=True),
    }


def _write_summaries(
    output_dir: Path,
    payload: dict[str, Any],
    raw_config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    written = {
        "json": output_dir / "json" / "plot-report.json",
        "markdown": output_dir / "markdown" / "plot-report.md",
        "config": output_dir / "json" / "plot-config.json",
    }
    write_json(written["json"], payload)
    write_json(written["config"], raw_config)
    _write_csv(output_dir, written, rows)
    ensure_dir(written["markdown"].parent)
    written["markdown"].write_text(_markdown(payload), encoding="utf-8")
    return written


def _write_csv(
    output_dir: Path, written: dict[str, Path], rows: list[dict[str, Any]]
) -> None:
    csv_path = output_dir / "csv" / "plot-report.csv"
    if rows:
        written["csv"] = csv_path
        write_csv(written["csv"], rows)
        return
    # Only this generator's fixed CSV output: do not leave a previous nonempty
    # summary masquerading as the current empty archive.
    csv_path.unlink(missing_ok=True)
