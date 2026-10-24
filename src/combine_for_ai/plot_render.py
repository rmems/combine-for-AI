"""Headless Matplotlib rendering. Importing this module requires the plots extra."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from combine_for_ai.jsonio import ensure_dir


_MAX_FOOTER_LINES = 12


def render_plot(path: Path, config: Any, figure: dict[str, Any]) -> None:
    cohort = figure["cohort"]
    metric = figure["metric"]
    plt.rcdefaults()
    observations = metric["observations"]
    footer = _footer_text(metric, config, figure["source_hash"], figure["caption"])
    footer_height = _footer_height(footer)
    fig, ax = plt.subplots(
        figsize=(config.width_inches, config.height_inches + footer_height)
    )
    try:
        _draw_observations(ax, observations)
        _decorate(ax, cohort, metric, config)
        _place_footer(fig, config, footer, footer_height)
        ensure_dir(path.parent)
        fig.savefig(
            path,
            dpi=config.dpi,
            metadata={"Software": "combine-for-ai headless reports v1"},
        )
    finally:
        plt.close(fig)


def _draw_observations(ax: Any, observations: list[dict[str, Any]]) -> None:
    colors = _draw_series(ax, observations)
    _draw_intervals(ax, observations, colors)


def _footer_text(
    metric: dict[str, Any], config: Any, source_hash: str, caption: str
) -> str:
    parts = [
        f"Definition: {metric['definition']}; scope: {metric['scope']}",
        caption,
        f"Source SHA256: {source_hash}; config: {config.schema_version}",
        "Counts, intervals and notes for every observation are in the report.",
    ]
    lines: list[str] = []
    for part in parts:
        lines.extend(_wrap(part, int(config.width_inches * 12)))
    if len(lines) <= _MAX_FOOTER_LINES:
        return "\n".join(lines)
    kept = lines[: _MAX_FOOTER_LINES - 1]
    return "\n".join([*kept, "…"])


def _wrap(part: str, width: int) -> list[str]:
    lines: list[str] = []
    for line in part.splitlines() or [""]:
        lines.extend(textwrap.wrap(line, width=width) or [""])
    return lines


def _footer_height(footer: str) -> float:
    return 0.16 * (footer.count("\n") + 1) + 0.25


def _draw_series(ax: Any, observations: list[dict[str, Any]]) -> dict[Any, str]:
    colors: dict[Any, str] = {}
    if _is_block(observations):
        _draw_blocks(ax, observations, colors)
    else:
        _draw_categories(ax, observations, colors)
    return colors


def _is_block(observations: list[dict[str, Any]]) -> bool:
    return all(obs["block_index"] is not None for obs in observations)


def _draw_blocks(
    ax: Any, observations: list[dict[str, Any]], colors: dict[Any, str]
) -> None:
    for series in dict.fromkeys(obs["series"] for obs in observations):
        points = _series_points(observations, series)
        (line,) = ax.plot(
            [obs["block_index"] for obs in points],
            [_point_value(obs) for obs in points],
            marker="o",
            label=series,
        )
        colors[series] = line.get_color()
    ax.set_xticks(sorted({obs["block_index"] for obs in observations}))
    ax.set_xlabel("Block index")
    ax.legend(loc="best")


def _series_points(observations: list[dict[str, Any]], series: str) -> list[dict[str, Any]]:
    matched = [obs for obs in observations if obs["series"] == series]
    return sorted(matched, key=lambda obs: obs["block_index"])


def _draw_categories(
    ax: Any, observations: list[dict[str, Any]], colors: dict[Any, str]
) -> None:
    for index, obs in enumerate(observations):
        if obs["value"] is None:
            continue
        (line,) = ax.plot(index, obs["value"], "o")
        colors[index] = line.get_color()
    ax.set_xticks(range(len(observations)), [_category_label(obs) for obs in observations])
    ax.set_xlabel("Prepared observations (not averaged)")


def _point_value(obs: dict[str, Any]) -> float:
    if obs["value"] is None:
        return float("nan")
    return float(obs["value"])


def _category_label(obs: dict[str, Any]) -> str:
    if obs["value"] is None:
        return f"{obs['series']}\n{obs['status']} — unavailable"
    return str(obs["series"])


def _draw_intervals(
    ax: Any, observations: list[dict[str, Any]], colors: dict[Any, str]
) -> None:
    block = _is_block(observations)
    style = {
        "block": block,
        "offsets": _series_offsets(observations, block),
        "colors": colors,
    }
    for index, obs in enumerate(observations):
        _draw_one_interval(ax, obs, index, style)


def _series_offsets(observations: list[dict[str, Any]], block: bool) -> dict[str, float]:
    if not block:
        return {}
    series = list(dict.fromkeys(obs["series"] for obs in observations))
    center = (len(series) - 1) / 2
    return {name: (index - center) * 0.12 for index, name in enumerate(series)}


def _draw_one_interval(
    ax: Any, obs: dict[str, Any], index: int, style: dict[str, Any]
) -> None:
    interval = obs["interval"]
    if interval is None:
        return
    block = style["block"]
    x = _x_position(obs, index, block, style["offsets"])
    color = _mark_color(obs, index, block, style["colors"])
    low, high = interval["low"], interval["high"]
    ax.vlines(x, low, high, color=color, linewidth=2)
    ax.plot([x, x], [low, high], "_", color=color, markersize=10)


def _x_position(
    obs: dict[str, Any], index: int, block: bool, offsets: dict[str, float]
) -> float:
    if not block:
        return float(index)
    return float(obs["block_index"]) + offsets[obs["series"]]


def _mark_color(
    obs: dict[str, Any], index: int, block: bool, colors: dict[Any, str]
) -> str:
    key = obs["series"] if block else index
    return colors.get(key, "#2463a0")


def _decorate(ax: Any, cohort: dict[str, Any], metric: dict[str, Any], config: Any) -> None:
    ax.set_ylabel(f"{metric['name']} [{metric['unit']}]")
    ax.set_title(_title(cohort, metric, config), fontsize=11)
    ax.grid(axis="y", alpha=0.25)


def _title(cohort: dict[str, Any], metric: dict[str, Any], config: Any) -> str:
    raw = (
        f"{cohort['workload_track']} · {metric['name']}\n"
        f"{cohort['execution_kind']} / {cohort['input_provenance']} / "
        f"{cohort['metric_provenance']}\n"
        f"{cohort['comparison_policy']} · cohort {cohort['id']}"
    )
    width = int(config.width_inches * 11)
    return "\n".join(textwrap.fill(line, width=width) for line in raw.splitlines())


def _place_footer(fig: Any, config: Any, footer: str, footer_height: float) -> None:
    total = config.height_inches + footer_height
    bottom = footer_height / total + 0.08
    top = 0.85
    if bottom >= top:
        bottom = top - 0.05
    fig.subplots_adjust(left=0.13, right=0.97, top=top, bottom=bottom)
    fig.text(0.04, 0.025, footer, fontsize=8, va="bottom", family="monospace")
