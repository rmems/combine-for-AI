from __future__ import annotations

from benchmarks.dataset_support import CATALOG, sample_path_for
from benchmarks.datasets import DatasetSpec
from benchmarks.wikitext.loader import (
    WikiText2Loader,
    concatenate_documents,
    map_wikitext_row,
)


def test_wikitext_sample_jsonl() -> None:
    dataset = WikiText2Loader().load(
        DatasetSpec(
            name="wikitext2",
            source="wikitext2",
            path=str(sample_path_for(CATALOG["wikitext2"])),
        )
    )
    assert len(dataset.records) == 2
    assert "Apollo" in dataset.records[0].prompt
    joined = concatenate_documents(dataset.records)
    assert "France" in joined
    assert dataset.records[1].reference == "Paris."


def test_map_wikitext_native_skips_empty() -> None:
    assert map_wikitext_row({"text": " \n"}) is None
    record = map_wikitext_row({"text": "  The capital city of France is Paris.  "})
    assert record is not None
    assert record.prompt == "The capital city of France is Paris."
    assert record.reference == record.prompt


def test_wikitext_hf_rows_skip_empty(tmp_path, monkeypatch) -> None:
    rows = [
        {"text": ""},
        {"text": "In 1969, the Apollo 11 mission landed on the moon."},
        {"text": "   "},
    ]
    monkeypatch.setattr(
        "benchmarks.dataset_mapped_cache.hf_load_dataset",
        lambda *args, **kwargs: rows,
    )
    dataset = WikiText2Loader().load(
        DatasetSpec(
            name="wikitext2",
            source="wikitext2",
            hf_id="wikitext",
            hf_subset="wikitext-2-raw-v1",
            cache_dir=str(tmp_path),
        )
    )
    assert len(dataset.records) == 1
    assert dataset.metadata["skipped"] == 2
    assert dataset.records[0].prompt.startswith("In 1969")
