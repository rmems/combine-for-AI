from __future__ import annotations

import re
from typing import Any

from benchmarks.dataset_jsonl import canonical_record
from benchmarks.dataset_support import MappedDatasetLoader, register_row_mapper
from benchmarks.dataset_types import DatasetRecord

COT_SUFFIX = "Let's think step by step."
_FINAL_ANSWER = re.compile(r"####\s*(.+)$")


def extract_gsm8k_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K gold or model completion."""
    for line in reversed(text.strip().splitlines()):
        match = _FINAL_ANSWER.search(line.strip())
        if match:
            return _normalize_gsm8k_answer(match.group(1))
    return _normalize_gsm8k_answer(text.strip())


def _normalize_gsm8k_answer(raw: str) -> str:
    return raw.strip().rstrip(".").replace(",", "")


def format_gsm8k_prompt(question: str) -> str:
    stripped = question.strip()
    if stripped.endswith(COT_SUFFIX):
        return stripped
    return f"{stripped}\n{COT_SUFFIX}"


def map_gsm8k_row(row: dict[str, Any]) -> DatasetRecord | None:
    """Map GSM8K using lm-eval ``gsm8k`` ``question`` / ``answer`` (``####``)."""
    record = canonical_record(row)
    if record is not None:
        return record

    question = row.get("question")
    answer = row.get("answer")
    if question is None:
        return None
    expected = extract_gsm8k_answer(str(answer or ""))
    if not expected:
        return None
    return DatasetRecord(
        prompt=format_gsm8k_prompt(str(question)),
        reference=expected,
    )


register_row_mapper("gsm8k", map_gsm8k_row)


class GSM8KLoader(MappedDatasetLoader):
    """Load GSM8K (lm-eval ``gsm8k``) with a CoT cue and extracted answers."""

    catalog_name = "gsm8k"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_gsm8k_row(row)
