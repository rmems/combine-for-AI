from __future__ import annotations

from typing import Any

from benchmarks.dataset_jsonl import canonical_record
from benchmarks.dataset_support import MappedDatasetLoader, register_row_mapper
from benchmarks.dataset_types import DatasetRecord


def map_wikitext_row(row: dict[str, Any]) -> DatasetRecord | None:
    """Map WikiText-2 using lm-eval ``wikitext`` ``text`` documents."""
    record = canonical_record(row)
    if record is not None:
        return record

    text = row.get("text")
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    return DatasetRecord(prompt=stripped, reference=stripped)


register_row_mapper("wikitext2", map_wikitext_row)


class WikiText2Loader(MappedDatasetLoader):
    """Load WikiText-2 (lm-eval ``wikitext``) for token-level perplexity."""

    catalog_name = "wikitext2"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_wikitext_row(row)


def concatenate_documents(records: list[DatasetRecord]) -> str:
    """Join non-empty documents for token-level perplexity scoring."""
    return "\n".join(record.prompt for record in records if record.prompt.strip())
