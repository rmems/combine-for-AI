"""Experiment-matrix cells, stable IDs, and definition fingerprints.

This is the resumability slice of the cross-model matrix runner: identity and
compatibility checks only. It does not produce comparison reports.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from benchmarks.datasets import DatasetSpec
from benchmarks.models import ModelSpec

MATRIX_IDENTITY_VERSION = "combine.matrix.cell.v1"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class MatrixError(Exception):
    """Base error for matrix definition and resume problems."""


class DuplicateCellIdError(MatrixError):
    """Two cells resolved to the same stable ID before execution."""


class IncompatibleMatrixError(MatrixError):
    """The on-disk journal was written for a different matrix definition."""


def canonical_dumps(value: Any) -> str:
    """Serialize ``value`` with stable key order and no insignificant whitespace."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_matrix_name(name: str) -> str:
    if not name or name in {".", ".."} or not _SAFE_NAME.fullmatch(name):
        raise MatrixError(
            f"invalid matrix_name {name!r}; use letters, digits, '.', '_' or '-'"
        )
    return name


@dataclass(frozen=True)
class CellIdentity:
    """Resolved coordinates that uniquely identify one matrix cell."""

    model: ModelSpec
    quantization: str
    dataset: DatasetSpec
    seed: int
    config_revision: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "v": MATRIX_IDENTITY_VERSION,
            "config_revision": self.config_revision,
            "dataset": {
                "hf_id": self.dataset.hf_id,
                "hf_subset": self.dataset.hf_subset,
                "max_samples": self.dataset.max_samples,
                "name": self.dataset.name,
                "path": self.dataset.path,
                "source": self.dataset.source,
                "split": self.dataset.split,
            },
            "model": {
                "backend": self.model.backend,
                "name": self.model.name,
                "revision": self.model.revision,
            },
            "quantization": self.quantization,
            "seed": self.seed,
        }

    def cell_id(self) -> str:
        return sha256_hex(canonical_dumps(self.to_canonical_dict()))

    def fingerprint(self) -> str:
        """Digest of the resolved identity; skip reuse requires this to match."""

        return sha256_hex("fp|" + canonical_dumps(self.to_canonical_dict()))

    def summary(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id(),
            "config_revision": self.config_revision,
            "dataset": self.dataset.name,
            "fingerprint": self.fingerprint(),
            "model": self.model.name,
            "quantization": self.quantization,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RetryPolicy:
    retry_failed: bool = False
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise MatrixError("max_attempts must be >= 1")

    @staticmethod
    def from_dict(raw: Mapping[str, Any] | None) -> "RetryPolicy":
        raw = raw or {}
        return RetryPolicy(
            retry_failed=bool(raw.get("retry_failed", False)),
            max_attempts=int(raw.get("max_attempts", 1)),
        )

    def allows_retry(self, attempt: int) -> bool:
        """``attempt`` is the number of completed failed tries so far."""

        return self.retry_failed and attempt < self.max_attempts


@dataclass(frozen=True)
class MatrixDefinition:
    name: str
    config_revision: str
    seed: int
    models: tuple[ModelSpec, ...]
    quantizations: tuple[str, ...]
    datasets: tuple[DatasetSpec, ...]
    retry: RetryPolicy
    cells: tuple[CellIdentity, ...]
    cell_by_id: dict[str, CellIdentity]
    fingerprint: str

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "MatrixDefinition":
        name = validate_matrix_name(str(raw.get("matrix_name", "")))
        config_revision = str(raw.get("config_revision", "1"))
        seed = int(raw.get("seed", 0))
        models, quantizations, datasets = _parse_axes(raw)
        retry_raw = raw.get("retry")
        retry = RetryPolicy.from_dict(retry_raw if isinstance(retry_raw, dict) else None)
        cells = expand_cells(models, quantizations, datasets, seed, config_revision)
        return MatrixDefinition(
            name=name,
            config_revision=config_revision,
            seed=seed,
            models=models,
            quantizations=quantizations,
            datasets=datasets,
            retry=retry,
            cells=cells,
            cell_by_id={cell.cell_id(): cell for cell in cells},
            fingerprint=definition_fingerprint(name, config_revision, cells),
        )


_T = TypeVar("_T")


def _require_axis(values: tuple[_T, ...], label: str) -> tuple[_T, ...]:
    if not values:
        raise MatrixError(f"matrix config must list at least one {label}")
    return values


def _parse_axes(
    raw: Mapping[str, Any],
) -> tuple[tuple[ModelSpec, ...], tuple[str, ...], tuple[DatasetSpec, ...]]:
    models = tuple(ModelSpec.from_dict(item) for item in raw.get("models") or [])
    quantizations = tuple(str(item) for item in raw.get("quantization") or [])
    datasets = tuple(DatasetSpec.from_dict(item) for item in raw.get("datasets") or [])
    return (
        _require_axis(models, "model"),
        _require_axis(quantizations, "quantization"),
        _require_axis(datasets, "dataset"),
    )


def expand_cells(
    models: tuple[ModelSpec, ...],
    quantizations: tuple[str, ...],
    datasets: tuple[DatasetSpec, ...],
    seed: int,
    config_revision: str,
) -> tuple[CellIdentity, ...]:
    cells: list[CellIdentity] = []
    seen: dict[str, CellIdentity] = {}
    for model in models:
        for quantization in quantizations:
            for dataset in datasets:
                identity = CellIdentity(
                    model=model,
                    quantization=quantization,
                    dataset=dataset,
                    seed=seed,
                    config_revision=config_revision,
                )
                cell_id = identity.cell_id()
                previous = seen.get(cell_id)
                if previous is not None:
                    raise DuplicateCellIdError(
                        "duplicate cell id "
                        f"{cell_id} for {identity.summary()} and {previous.summary()}"
                    )
                seen[cell_id] = identity
                cells.append(identity)
    return tuple(cells)


def definition_fingerprint(
    name: str,
    config_revision: str,
    cells: tuple[CellIdentity, ...] | list[CellIdentity],
) -> str:
    payload = {
        "cells": [cell.to_canonical_dict() for cell in cells],
        "config_revision": config_revision,
        "matrix_name": name,
    }
    return sha256_hex(canonical_dumps(payload))


def assert_compatible(journal_fingerprint: str, definition: MatrixDefinition) -> None:
    if journal_fingerprint == definition.fingerprint:
        return
    raise IncompatibleMatrixError(
        "matrix definition does not match the journal: "
        f"journal={journal_fingerprint} current={definition.fingerprint}"
    )
