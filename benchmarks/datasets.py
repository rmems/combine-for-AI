from __future__ import annotations

from collections.abc import Iterable

from benchmarks.arc_easy.loader import ARCEasyLoader
from benchmarks.dataset_support import HuggingFaceDatasetLoader, JsonlDatasetLoader
from benchmarks.dataset_types import (
    DatasetLoader,
    DatasetRecord,
    DatasetSpec,
    LoadedDataset,
)
from benchmarks.gsm8k.loader import GSM8KLoader
from benchmarks.hellaswag.loader import HellaSwagLoader
from benchmarks.lambada.loader import LAMBADALoader
from benchmarks.piqa.loader import PIQALoader
from benchmarks.wikitext.loader import WikiText2Loader


class DatasetRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, DatasetLoader] = {}
        self._named: list[str] = []

    def register(self, source: str, loader: DatasetLoader) -> None:
        self._loaders[source] = loader

    def register_named(
        self,
        name: str,
        loader: DatasetLoader,
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        self.register(name, loader)
        self._named.append(name)
        for alias in aliases:
            self.register(alias, loader)

    def loader_for(self, source: str) -> DatasetLoader:
        if source not in self._loaders:
            raise ValueError(f"dataset source '{source}' is not registered")
        return self._loaders[source]

    def available_sources(self) -> Iterable[str]:
        return sorted(self._loaders.keys())

    def named_datasets(self) -> tuple[str, ...]:
        return tuple(self._named)

    def iter_named_datasets(self) -> Iterable[tuple[str, DatasetLoader]]:
        for name in self._named:
            yield name, self._loaders[name]


def default_dataset_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register("jsonl", JsonlDatasetLoader())
    registry.register("hf", HuggingFaceDatasetLoader())
    registry.register_named("lambada", LAMBADALoader())
    registry.register_named("hellaswag", HellaSwagLoader())
    registry.register_named("wikitext2", WikiText2Loader(), aliases=("wikitext",))
    registry.register_named("gsm8k", GSM8KLoader())
    registry.register_named("piqa", PIQALoader())
    registry.register_named("arc_easy", ARCEasyLoader(), aliases=("arc-easy",))
    return registry


__all__ = [
    "DatasetLoader",
    "DatasetRecord",
    "DatasetRegistry",
    "DatasetSpec",
    "HuggingFaceDatasetLoader",
    "JsonlDatasetLoader",
    "LoadedDataset",
    "default_dataset_registry",
]
