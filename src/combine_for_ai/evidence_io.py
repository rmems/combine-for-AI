"""JSON, CSV and Markdown round trips for evidence records.

CSV and Markdown share one flat shape (one row per metric, record fields
repeated) so a record survives every format with raw hashes, units and
independent provenance intact. Writers reuse ``write_json`` / ``write_csv``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from combine_for_ai.csvio import CSV_FORMULA_PREFIXES, write_csv
from combine_for_ai.evidence import (
    EvidenceError,
    EvidenceRecord,
    load_evidence,
    parse_evidence,
)
from combine_for_ai.jsonio import write_json

_RECORD_COLUMNS = (
    "schema",
    "schema_version",
    "record_id",
    "workload_track",
    "architecture",
    "execution_kind",
    "data_provenance",
    "requested_data_provenance",
    "capabilities",
    "versions",
    "seed",
    "budget",
    "environment_identity",
    "record_source_artifact_sha256",
    "notes",
)
_METRIC_COLUMNS = (
    "metric_name",
    "native_name",
    "unit",
    "definition",
    "value",
    "status",
    "provenance",
    "source_artifact_sha256",
    "reason",
)
EVIDENCE_COLUMNS = _RECORD_COLUMNS + _METRIC_COLUMNS

_JSON_COLUMNS = ("capabilities", "versions", "budget")


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)  # shortest string that round-trips exactly
    return str(value)


def record_to_rows(record: EvidenceRecord) -> list[dict[str, str]]:
    payload = record.to_payload()
    base = {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "record_id": payload["record_id"],
        "workload_track": payload["workload_track"],
        "architecture": payload["architecture"],
        "execution_kind": payload["execution_kind"],
        "data_provenance": payload["data_provenance"],
        "requested_data_provenance": payload["requested_data_provenance"],
        "capabilities": json.dumps(payload["capabilities"]),
        "versions": json.dumps(payload["versions"], sort_keys=True),
        "seed": payload["seed"],
        "budget": json.dumps(payload["budget"], sort_keys=True),
        "environment_identity": payload["environment_identity"],
        "record_source_artifact_sha256": payload["source_artifact_sha256"],
        "notes": payload["notes"],
    }
    base_cells = {key: _cell(value) for key, value in base.items()}
    metrics = payload["metrics"] or [None]
    rows: list[dict[str, str]] = []
    for metric in metrics:
        row = dict(base_cells)
        metric = metric or {}
        row.update(
            {
                "metric_name": _cell(metric.get("name")),
                "native_name": _cell(metric.get("native_name")),
                "unit": _cell(metric.get("unit")),
                "definition": _cell(metric.get("definition")),
                "value": _cell(metric.get("value")),
                "status": _cell(metric.get("status")),
                "provenance": _cell(metric.get("provenance")),
                "source_artifact_sha256": _cell(metric.get("source_artifact_sha256")),
                "reason": _cell(metric.get("reason")),
            }
        )
        rows.append(row)
    return rows


def _float_cell(text: str, metric: str) -> float | None:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise EvidenceError(f"metric '{metric}': malformed value {text!r}") from exc


def _opt(text: str) -> str | None:
    return text if text != "" else None


def _group_rows(rows: Iterable[Mapping[str, str]]) -> list[list[Mapping[str, str]]]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        missing = [col for col in EVIDENCE_COLUMNS if col not in row]
        if missing:
            raise EvidenceError(f"evidence row missing columns: {', '.join(missing)}")
        grouped.setdefault(row["record_id"], []).append(row)
    return list(grouped.values())


def _check_consistent(group: list[Mapping[str, str]]) -> None:
    first = group[0]
    for other in group[1:]:
        if any(other[col] != first[col] for col in _RECORD_COLUMNS):
            raise EvidenceError(f"inconsistent record fields for '{first['record_id']}'")


def _metric_from_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "name": row["metric_name"],
        "native_name": _opt(row["native_name"]),
        "unit": _opt(row["unit"]),
        "definition": _opt(row["definition"]),
        "value": _float_cell(row["value"], row["metric_name"]),
        "status": row["status"],
        "provenance": row["provenance"],
        "source_artifact_sha256": _opt(row["source_artifact_sha256"]),
        "reason": _opt(row["reason"]),
    }


def _record_payload(first: Mapping[str, str], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        payload: dict[str, Any] = {
            "schema": first["schema"],
            "schema_version": int(first["schema_version"]),
            "record_id": first["record_id"],
            "workload_track": _opt(first["workload_track"]),
            "architecture": _opt(first["architecture"]),
            "execution_kind": first["execution_kind"],
            "data_provenance": first["data_provenance"],
            "requested_data_provenance": _opt(first["requested_data_provenance"]),
            "seed": int(first["seed"]) if first["seed"] != "" else None,
            "environment_identity": _opt(first["environment_identity"]),
            "source_artifact_sha256": _opt(first["record_source_artifact_sha256"]),
            "notes": _opt(first["notes"]),
            "metrics": metrics,
        }
        for column in _JSON_COLUMNS:
            payload[column] = json.loads(first[column])
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise EvidenceError(f"malformed evidence row: {exc}") from exc
    return payload


def records_from_rows(rows: Iterable[Mapping[str, str]]) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for group in _group_rows(rows):
        _check_consistent(group)
        metrics = [_metric_from_row(row) for row in group if row["metric_name"] != ""]
        records.append(parse_evidence(_record_payload(group[0], metrics)))
    return records


# --- JSON -----------------------------------------------------------------


def write_evidence_json(path: Path, records: Sequence[EvidenceRecord]) -> None:
    write_json(path, {"records": [record.to_payload() for record in records]})


def read_evidence_json(path: Path) -> list[EvidenceRecord]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("records") if isinstance(payload, dict) and "records" in payload else [payload]
    if not isinstance(items, list):
        raise EvidenceError("evidence JSON must hold a record or a 'records' list")
    return [load_evidence(item) for item in items]


# --- CSV ------------------------------------------------------------------


def write_evidence_csv(path: Path, records: Sequence[EvidenceRecord]) -> None:
    rows = [row for record in records for row in record_to_rows(record)]
    if not rows:
        raise EvidenceError("no evidence rows to write")
    # write_csv prefixes ``'`` to formula-looking cells. Escape cells that already
    # start with ``'`` the same way so _unguard can tell the two cases apart.
    write_csv(path, [{k: _escape_apostrophe(v) for k, v in row.items()} for row in rows])


def read_evidence_csv(path: Path) -> list[EvidenceRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {key: _unguard(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    return records_from_rows(rows)


def _escape_apostrophe(value: str) -> str:
    return f"'{value}" if value.startswith("'") else value


def _unguard(value: str | None) -> str:
    """Undo the leading-apostrophe guards (formula prefixes and escaped ``'``)."""
    if value is None:
        return ""
    if len(value) > 1 and value[0] == "'" and (value[1] in CSV_FORMULA_PREFIXES or value[1] == "'"):
        return value[1:]
    return value


# --- Markdown -------------------------------------------------------------


def _md_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _md_split(line: str) -> list[str]:
    cells: list[str] = []
    buf: list[str] = []
    chars = iter(line.strip()[1:])  # drop the leading pipe
    for ch in chars:
        if ch == "\\":
            nxt = next(chars, "")
            buf.append({"n": "\n", "r": "\r"}.get(nxt, nxt))
        elif ch == "|":
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    return [c[1:-1] if len(c) >= 2 else c for c in cells]


def render_evidence_markdown(records: Sequence[EvidenceRecord]) -> str:
    lines = ["# Evidence records", ""]
    lines.append("| " + " | ".join(EVIDENCE_COLUMNS) + " |")
    lines.append("|" + "|".join(" --- " for _ in EVIDENCE_COLUMNS) + "|")
    for record in records:
        for row in record_to_rows(record):
            lines.append("|" + "|".join(f" {_md_escape(row[c])} " for c in EVIDENCE_COLUMNS) + "|")
    return "\n".join(lines) + "\n"


def parse_evidence_markdown(text: str) -> list[EvidenceRecord]:
    # split("\n") only: _md_escape guarantees no raw \n, but other separators splitlines() honours may appear in cells.
    table = [line.rstrip("\r") for line in text.split("\n") if line.startswith("|")]
    if len(table) < 2:
        raise EvidenceError("no evidence table found in markdown")
    header = [c.strip() for c in _md_split(table[0])]
    if tuple(header) != EVIDENCE_COLUMNS:
        raise EvidenceError("markdown evidence table has unexpected columns")
    rows = []
    for line in table[2:]:
        cells = _md_split(line)
        if len(cells) != len(header):
            raise EvidenceError("markdown evidence row has wrong cell count")
        rows.append(dict(zip(header, cells, strict=True)))
    return records_from_rows(rows)


def write_evidence_markdown(path: Path, records: Sequence[EvidenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_evidence_markdown(records), encoding="utf-8")


def read_evidence_markdown(path: Path) -> list[EvidenceRecord]:
    return parse_evidence_markdown(path.read_text(encoding="utf-8"))


__all__ = [
    "EVIDENCE_COLUMNS",
    "parse_evidence_markdown",
    "read_evidence_csv",
    "read_evidence_json",
    "read_evidence_markdown",
    "record_to_rows",
    "records_from_rows",
    "render_evidence_markdown",
    "write_evidence_csv",
    "write_evidence_json",
    "write_evidence_markdown",
]
