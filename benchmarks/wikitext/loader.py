from __future__ import annotations

from typing import Any

from benchmarks.dataset_jsonl import canonical_record
from benchmarks.dataset_support import MappedDatasetLoader, register_row_mapper
from benchmarks.dataset_types import DatasetRecord


def map_wikitext_row(row: dict[str, Any]) -> DatasetRecord | None:
    """Map WikiText-2 using lm-eval ``wikitext`` ``text`` documents.

    Raw Hub rows keep their original ``text`` (including blank separators) as
    the perplexity prompt and do not copy it into ``reference``, so the runner
    scores token-level NLL rather than exact document reproduction.
    """
    record = canonical_record(row)
    if record is not None:
        return record

    if "text" not in row or row["text"] is None:
        return None
    return DatasetRecord(prompt=str(row["text"]))


register_row_mapper("wikitext2", map_wikitext_row)


class WikiText2Loader(MappedDatasetLoader):
    """Load WikiText-2 (lm-eval ``wikitext``) for token-level perplexity."""

    catalog_name = "wikitext2"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_wikitext_row(row)


def concatenate_documents(records: list[DatasetRecord]) -> str:
    """Concatenate the raw WikiText stream without inserting extra separators."""
    return "".join(record.prompt for record in records)
