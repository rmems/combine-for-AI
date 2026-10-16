"""Regenerate headless reports from archived observations, never statistics."""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from benchmarks.reporting import write_csv
from combine_for_ai.jsonio import ensure_dir, write_json
from combine_for_ai.report_evidence import (
    RenderingModel,
    load_report_evidence,
    read_json,
)


class PlotConfig(RenderingModel):
    schema_version: Literal["combine.plot-config.v1"]
    width_inches: float = Field(default=9, ge=6, le=20)
    height_inches: float = Field(default=5, ge=4, le=20)
    dpi: int = Field(default=120, ge=72, le=300)


def _interval_text(interval: dict[str, Any] | None) -> str:
    if interval is None:
        return "unavailable (no prepared uncertainty)"
    return (
        f"[{interval['low']:g}, {interval['high']:g}]; "
        f"{interval['confidence_level'] * 100:g}% {interval['method']}; "
        f"unit={interval['experimental_unit']}; "
        f"{interval['library']} {interval['library_version']}; {interval['definition']}"
    )


def _counts_text(counts: dict[str, int]) -> str:
    return (
        ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        if counts
        else "unavailable"
    )


def _caption(cohort: dict[str, Any], metric: dict[str, Any]) -> str:
    statuses = Counter(o["status"] for o in metric["observations"])
    return (
        f"n={cohort['sample_count'] if cohort['sample_count'] is not None else 'unknown'}; "
        f"repeats={cohort['repeat_count'] if cohort['repeat_count'] is not None else 'unknown'}; "
        f"seed policy: {cohort['seed_policy']}\n"
        f"Observation statuses: {_counts_text(dict(statuses))} (not sample denominators)"
    )


def _render_plot(cohort, metric, path, config, source_hash):
    # Imported only for the optional plotting step, never by summary-only users.
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plt.rcdefaults()
    observations = metric["observations"]
    footer = [
        f"Definition: {metric['definition']}; scope: {metric['scope']}",
        _caption(cohort, metric),
    ]
    for obs in observations:
        footer.append(
            f"{obs['series']}"
            + (f" block={obs['block_index']}" if obs["block_index"] is not None else "")
            + f": counts: {_counts_text(obs['counts'])}; interval: {_interval_text(obs['interval'])}"
            + (f"; {obs['notes']}" if obs["notes"] else "")
        )
    footer.append(f"Source SHA256: {source_hash}; config: {config.schema_version}")
    wrapped = "\n".join(
        textwrap.fill(line, width=int(config.width_inches * 12))
        for part in footer
        for line in part.splitlines()
    )
    footer_height = 0.16 * (wrapped.count("\n") + 1) + 0.25
    fig, ax = plt.subplots(
        figsize=(config.width_inches, config.height_inches + footer_height)
    )
    try:
        is_block = all(o["block_index"] is not None for o in observations)
        if is_block:
            for series in dict.fromkeys(o["series"] for o in observations):
                points = sorted(
                    (o for o in observations if o["series"] == series),
                    key=lambda o: o["block_index"],
                )
                # NaN breaks a missing segment; it never substitutes zero.
                ax.plot(
                    [o["block_index"] for o in points],
                    [
                        o["value"] if o["value"] is not None else float("nan")
                        for o in points
                    ],
                    marker="o",
                    label=series,
                )
            ax.set_xticks(sorted({o["block_index"] for o in observations}))
            ax.set_xlabel("Block index")
            ax.legend(loc="best")
        else:
            for index, obs in enumerate(observations):
                if obs["value"] is not None:
                    ax.plot(index, obs["value"], "o", color="#2463a0")
            ax.set_xticks(
                range(len(observations)),
                [
                    o["series"]
                    + (f"\n{o['status']} — unavailable" if o["value"] is None else "")
                    for o in observations
                ],
            )
            ax.set_xlabel("Prepared observations (not averaged)")
        # Draw upstream interval endpoints directly: a valid interval need not
        # contain the point estimate. No errorbar arithmetic/statistics here.
        for index, obs in enumerate(observations):
            if obs["interval"] is not None:
                x = obs["block_index"] if is_block else index
                low, high = obs["interval"]["low"], obs["interval"]["high"]
                ax.vlines(x, low, high, color="#2463a0", linewidth=2)
                ax.plot([x, x], [low, high], "_", color="#2463a0", markersize=10)
        ax.set_ylabel(f"{metric['name']} [{metric['unit']}]")
        title = (
            f"{cohort['workload_track']} · {metric['name']}\n"
            f"{cohort['execution_kind']} / {cohort['input_provenance']} / {cohort['metric_provenance']}\n"
            f"{cohort['comparison_policy']} · cohort {cohort['id']}"
        )
        ax.set_title(
            "\n".join(
                textwrap.fill(line, width=int(config.width_inches * 11))
                for line in title.splitlines()
            ),
            fontsize=11,
        )
        ax.grid(axis="y", alpha=0.25)
        fig.subplots_adjust(
            left=0.13,
            right=0.97,
            top=0.85,
            bottom=footer_height / (config.height_inches + footer_height) + 0.12,
        )
        fig.text(0.04, 0.025, wrapped, fontsize=8, va="bottom", family="monospace")
        ensure_dir(path.parent)
        fig.savefig(
            path,
            dpi=config.dpi,
            metadata={"Software": "combine-for-ai headless reports v1"},
        )
    finally:
        plt.close(fig)


def _cell(value: Any) -> str:
    text = "unavailable" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Archived per-track reports",
        "",
        "No averaging or uncertainty estimation is performed here. Availability statuses are not pass/fail qualification gates.",
        "Fixtures, mock/header-only execution, imports, measured execution and calculated-proxy metrics remain distinct. Operation/energy proxies are not device energy measurements.",
        "",
        f"Plot configuration: `{payload['plot_config']['schema_version']}`; SHA256 `{payload['plot_config_sha256']}`.",
        "",
        "## Sources",
        "",
    ]
    for source in payload["sources"]:
        lines.extend(
            [
                f"- {_cell(source['path'])}: SHA256 `{source['sha256']}`",
                f"  Run/link metadata: {_cell(json.dumps(source['run'], sort_keys=True))}",
                f"  Source provenance: {_cell(json.dumps(source['provenance'], sort_keys=True))}",
            ]
        )
    if not payload["cohorts"]:
        lines.extend(["", "No observations; no plots generated.", ""])
    for cohort in payload["cohorts"]:
        lines.extend(
            [
                "",
                f"## {_cell(cohort['workload_track'])} — cohort {cohort['id']}",
                "",
                f"Evidence: {_cell(cohort['execution_kind'])} / {_cell(cohort['input_provenance'])} / {_cell(cohort['metric_provenance'])}.",
                f"Comparison policy: {_cell(cohort['comparison_policy'])}. Source: {_cell(cohort['source'])}.",
                f"Context: {_cell(json.dumps(cohort['context'], sort_keys=True))}",
                "",
            ]
        )
        if not cohort["metrics"]:
            lines.append(
                "No metrics supplied; availability and failure counts unknown."
            )
        for metric in cohort["metrics"]:
            lines.extend(
                [
                    f"### {_cell(metric['name'])} [{_cell(metric['unit'])}]",
                    "",
                    f"Definition: {_cell(metric['definition'])}. Scope: {_cell(metric['scope'])}.",
                    _cell(_caption(cohort, metric)),
                    "",
                    "| Series | Block | Value | Status | Upstream counts (denominator policy in definition) | Prepared interval | Notes | Raw artifacts |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for obs in metric["observations"]:
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
                lines.append("| " + " | ".join(_cell(v) for v in cells) + " |")
            if metric.get("plot"):
                lines.extend(
                    ["", f"![{_cell(metric['name'])}](../{metric['plot']})", ""]
                )
            else:
                lines.extend(
                    ["", "Plot not generated (disabled or all values unavailable).", ""]
                )
    return "\n".join(lines) + "\n"


def generate_reports(
    paths: list[Path],
    output_dir: Path,
    config_path: Path,
    *,
    plots: bool = True,
) -> dict[str, Path]:
    """Validate all inputs before writing reproducible reports under output_dir."""
    raw_config, config_hash = read_json(Path(config_path))
    config = PlotConfig.model_validate(raw_config)
    sources, cohorts = load_report_evidence(paths)
    output_dir = Path(output_dir)
    if plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "Plotting requires optional dependencies: uv sync --extra plots; or use --no-plots"
            ) from exc
    payload = {
        "schema_version": "combine.plot-report.v1",
        "plot_config": config.model_dump(),
        "plot_config_sha256": config_hash,
        "sources": sources,
        "cohorts": cohorts,
    }
    rows = []
    for cohort in cohorts:
        source_hash = next(
            s["sha256"] for s in sources if s["path"] == cohort["source"]
        )
        for index, metric in enumerate(cohort["metrics"]):
            if plots and any(o["value"] is not None for o in metric["observations"]):
                relative = f"plots/{cohort['id']}-{index:02d}.png"
                _render_plot(cohort, metric, output_dir / relative, config, source_hash)
                metric["plot"] = relative
            for obs in metric["observations"]:
                rows.append(
                    {
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
                        "source_artifacts": json.dumps(
                            obs["source_artifacts"], sort_keys=True
                        ),
                    }
                )
    written = {
        "json": output_dir / "json" / "plot-report.json",
        "markdown": output_dir / "markdown" / "plot-report.md",
        "config": output_dir / "json" / "plot-config.json",
    }
    write_json(written["json"], payload)
    write_json(written["config"], raw_config)
    if rows:
        written["csv"] = output_dir / "csv" / "plot-report.csv"
        write_csv(written["csv"], rows)
    else:
        # Only this generator's fixed CSV output: do not leave a previous
        # nonempty summary masquerading as the current empty archive.
        (output_dir / "csv" / "plot-report.csv").unlink(missing_ok=True)
    ensure_dir(written["markdown"].parent)
    written["markdown"].write_text(_markdown(payload), encoding="utf-8")
    return written
