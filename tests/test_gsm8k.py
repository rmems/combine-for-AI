from __future__ import annotations

from benchmarks.dataset_support import CATALOG, sample_path_for
from benchmarks.datasets import DatasetSpec
from benchmarks.gsm8k.loader import (
    COT_SUFFIX,
    GSM8KLoader,
    extract_gsm8k_answer,
    format_gsm8k_prompt,
    map_gsm8k_row,
)


def test_gsm8k_sample_jsonl() -> None:
    dataset = GSM8KLoader().load(
        DatasetSpec(
            name="gsm8k",
            source="gsm8k",
            path=str(sample_path_for(CATALOG["gsm8k"])),
        )
    )
    assert len(dataset.records) == 2
    assert dataset.records[0].reference == "72"
    assert dataset.records[1].reference == "10"
    assert dataset.records[0].prompt.endswith(COT_SUFFIX)


def test_extract_gsm8k_answer() -> None:
    gold = "Natalia sold 48+24 = 72 clips.\n#### 72"
    assert extract_gsm8k_answer(gold) == "72"
    assert extract_gsm8k_answer("#### 1,000") == "1000"
    assert extract_gsm8k_answer("plain 42") == "plain 42"


def test_format_gsm8k_prompt_is_idempotent() -> None:
    question = "How many clips?"
    once = format_gsm8k_prompt(question)
    assert once.endswith(COT_SUFFIX)
    assert format_gsm8k_prompt(once) == once


def test_map_gsm8k_canonical_passthrough() -> None:
    record = map_gsm8k_row(
        {"prompt": f"Q\n{COT_SUFFIX}", "reference": "7"}
    )
    assert record is not None
    assert record.reference == "7"
