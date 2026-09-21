from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from benchmarks.datasets import DatasetRecord, DatasetSpec, JsonlDatasetLoader, LoadedDataset

REPO_ROOT = Path(__file__).resolve().parent.parent


class TaskKind(str, Enum):
    CLASSIFICATION = "classification"
    CLOZE = "cloze"
    PERPLEXITY = "perplexity"
    EXACT_MATCH_MATH = "exact_match_math"


class DatasetCaseError(ValueError):
    """Invalid dataset-case shape or batch invariant."""

    def __init__(
        self,
        message: str,
        *,
        dataset: str | None = None,
        example_id: str | None = None,
        expected: Any = None,
        observed: Any = None,
    ) -> None:
        self.dataset = dataset
        self.example_id = example_id
        self.expected = expected
        self.observed = observed
        parts = [message]
        if dataset is not None or example_id is not None:
            parts.append(f"dataset={dataset!r}, example_id={example_id!r}")
        if expected is not None or observed is not None:
            parts.append(f"expected={expected!r}, observed={observed!r}")
        super().__init__("; ".join(parts))


def _case_label(dataset: str, example_id: str) -> str:
    return f"dataset={dataset!r}, example_id={example_id!r}"


def _reject_blank_choice(choices: list[str], *, dataset: str, example_id: str) -> None:
    label = _case_label(dataset, example_id)
    for index, choice in enumerate(choices):
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError(
                f"malformed choices ({label}): choice {index} must be "
                "a non-empty string"
            )


def _reject_classification_shape(
    *,
    dataset: str,
    example_id: str,
    choices: list[str] | None,
    answer_index: int | None,
    expected: str | None,
) -> None:
    label = _case_label(dataset, example_id)
    if choices is None or len(choices) < 2:
        raise ValueError(
            f"malformed choices ({label}): expected a list of at least "
            "2 non-empty strings"
        )
    _reject_blank_choice(choices, dataset=dataset, example_id=example_id)
    if answer_index is None:
        raise ValueError(f"missing target ({label}): answer_index is required")
    if not 0 <= answer_index < len(choices):
        raise ValueError(
            f"malformed choices ({label}): answer_index "
            f"{answer_index} is out of range for {len(choices)} choices"
        )
    if expected is not None and str(expected).strip():
        raise ValueError(
            f"malformed classification target ({label}): "
            "expected/reference must not be set; use choices and answer_index"
        )


def _reject_open_ended_shape(
    *,
    dataset: str,
    example_id: str,
    task: TaskKind,
    expected: str | None,
    choices: list[str] | None,
    answer_index: int | None,
) -> None:
    label = _case_label(dataset, example_id)
    if choices is not None:
        raise ValueError(
            f"malformed choices ({label}): choices are only valid for "
            "classification tasks"
        )
    if answer_index is not None:
        raise ValueError(
            f"malformed choices ({label}): answer_index is only valid for "
            "classification tasks"
        )
    if expected is None or not str(expected).strip():
        raise ValueError(
            f"missing target ({label}): {task.value} requires a "
            "non-empty expected answer"
        )


class DatasetCase(BaseModel):
    """Canonical scored example shared by every dataset family.

    Loader rows stay as ``DatasetRecord``. Convert with ``record_to_case`` /
    ``case_to_record`` / ``cases_from_loaded``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    task: TaskKind
    expected: str | None = None
    choices: list[str] | None = None
    answer_index: int | None = None
    source: str = "jsonl"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("example_id", "dataset", "split", "source", mode="before")
    @classmethod
    def _strip_identity(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("answer_index", mode="before")
    @classmethod
    def _reject_bool_answer_index(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("answer_index must be an integer, not a boolean")
        return value

    @field_validator("prompt")
    @classmethod
    def _reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace characters")
        return value

    @property
    def target(self) -> str:
        if self.task is TaskKind.CLASSIFICATION:
            if self.choices is None or self.answer_index is None:
                raise DatasetCaseError(
                    "classification case is missing a target",
                    dataset=self.dataset,
                    example_id=self.example_id,
                    expected="choices[answer_index]",
                    observed=None,
                )
            if not 0 <= self.answer_index < len(self.choices):
                raise DatasetCaseError(
                    "answer_index out of bounds",
                    dataset=self.dataset,
                    example_id=self.example_id,
                    expected=f"index in 0..{len(self.choices) - 1}",
                    observed=self.answer_index,
                )
            return self.choices[self.answer_index]
        if self.expected is None:
            raise DatasetCaseError(
                "case is missing a target",
                dataset=self.dataset,
                example_id=self.example_id,
                expected="non-empty expected",
                observed=self.expected,
            )
        return self.expected

    @model_validator(mode="after")
    def _validate_task_shape(self) -> DatasetCase:
        if self.task is TaskKind.CLASSIFICATION:
            _reject_classification_shape(
                dataset=self.dataset,
                example_id=self.example_id,
                choices=self.choices,
                answer_index=self.answer_index,
                expected=self.expected,
            )
        else:
            _reject_open_ended_shape(
                dataset=self.dataset,
                example_id=self.example_id,
                task=self.task,
                expected=self.expected,
                choices=self.choices,
                answer_index=self.answer_index,
            )
        return self


@dataclass(frozen=True)
class FamilySpec:
    name: str
    task: TaskKind
    sample_relpath: str
    hf_id: str
    hf_subset: str | None = None
    aliases: tuple[str, ...] = ()


FAMILY_SPECS: dict[str, FamilySpec] = {
    "lambada": FamilySpec(
        name="lambada",
        task=TaskKind.CLOZE,
        sample_relpath="configs/datasets/lambada.sample.jsonl",
        hf_id="EleutherAI/lambada_openai",
        aliases=("lambada",),
    ),
    "hellaswag": FamilySpec(
        name="hellaswag",
        task=TaskKind.CLASSIFICATION,
        sample_relpath="configs/datasets/hellaswag.sample.jsonl",
        hf_id="Rowan/hellaswag",
        aliases=("hellaswag", "hella_swag"),
    ),
    "wikitext2": FamilySpec(
        name="wikitext2",
        task=TaskKind.PERPLEXITY,
        sample_relpath="configs/datasets/wikitext2.sample.jsonl",
        hf_id="Salesforce/wikitext",
        hf_subset="wikitext-2-raw-v1",
        aliases=("wikitext2", "wikitext-2", "wikitext_2", "wiki-text-2"),
    ),
    "gsm8k": FamilySpec(
        name="gsm8k",
        task=TaskKind.EXACT_MATCH_MATH,
        sample_relpath="configs/datasets/gsm8k.sample.jsonl",
        hf_id="openai/gsm8k",
        hf_subset="main",
        aliases=("gsm8k", "gsm-8k"),
    ),
    "piqa": FamilySpec(
        name="piqa",
        task=TaskKind.CLASSIFICATION,
        sample_relpath="configs/datasets/piqa.sample.jsonl",
        hf_id="ybisk/piqa",
        aliases=("piqa",),
    ),
    "arc_easy": FamilySpec(
        name="arc_easy",
        task=TaskKind.CLASSIFICATION,
        sample_relpath="configs/datasets/arc_easy.sample.jsonl",
        hf_id="allenai/ai2_arc",
        hf_subset="ARC-Easy",
        aliases=("arc_easy", "arc-easy", "arceasy", "arc"),
    ),
}


def _normalize_family_key(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "_")


_ALIAS_TO_FAMILY: dict[str, str] = {}
for _spec in FAMILY_SPECS.values():
    for _alias in (_spec.name, *_spec.aliases):
        _ALIAS_TO_FAMILY[_normalize_family_key(_alias)] = _spec.name


def normalize_family_name(name: str) -> str:
    key = _normalize_family_key(name)
    if key not in _ALIAS_TO_FAMILY:
        known = ", ".join(sorted(FAMILY_SPECS))
        raise DatasetCaseError(
            f"unknown dataset family {name!r}; known families: {known}",
            dataset=name,
            example_id=None,
        )
    return _ALIAS_TO_FAMILY[key]


def family_spec(name: str) -> FamilySpec:
    return FAMILY_SPECS[normalize_family_name(name)]


def available_families() -> tuple[str, ...]:
    return tuple(FAMILY_SPECS)


def stable_example_id(dataset: str, split: str, index: int, explicit: str | None = None) -> str:
    if explicit:
        example_id = explicit.strip()
        if example_id:
            return example_id
    return f"{dataset}:{split}:{index:04d}"


def resolve_example_id(
    dataset: str,
    split: str,
    index: int,
    explicit: str | None,
) -> str:
    """Use fixture IDs when consistent; regenerate when the split override disagrees."""
    if explicit is not None:
        candidate = str(explicit).strip()
        if candidate:
            parts = candidate.split(":")
            if (
                len(parts) >= 3
                and parts[0] == dataset
                and parts[1] != split
                and parts[2].isdigit()
            ):
                return stable_example_id(dataset, split, index, None)
            return candidate
    return stable_example_id(dataset, split, index, None)


def parse_case(raw: Mapping[str, Any]) -> DatasetCase:
    payload = dict(raw)
    try:
        return DatasetCase.model_validate(payload)
    except ValidationError as exc:
        raise DatasetCaseError(
            f"invalid case: {exc}",
            dataset=payload.get("dataset"),
            example_id=payload.get("example_id"),
        ) from exc


def validate_case_batch(cases: Iterable[DatasetCase]) -> list[DatasetCase]:
    materialized = list(cases)
    seen: dict[str, DatasetCase] = {}
    for case in materialized:
        prior = seen.get(case.example_id)
        if prior is not None:
            raise DatasetCaseError(
                "duplicate example ID",
                dataset=case.dataset,
                example_id=case.example_id,
                expected="unique example_id",
                observed=case.example_id,
            )
        seen[case.example_id] = case
    return materialized


def infer_task(record: DatasetRecord, dataset: str | None = None) -> TaskKind:
    if dataset is not None:
        try:
            return family_spec(dataset).task
        except DatasetCaseError:
            pass
    if record.is_multiple_choice:
        return TaskKind.CLASSIFICATION
    return TaskKind.CLOZE


def record_to_case(
    record: DatasetRecord,
    *,
    dataset: str,
    split: str,
    index: int = 0,
    example_id: str | None = None,
    task: TaskKind | None = None,
    source: str = "jsonl",
    metadata: Mapping[str, Any] | None = None,
) -> DatasetCase:
    """Adapt a loader ``DatasetRecord`` to the canonical case shape."""
    resolved_task = task or infer_task(record, dataset)
    family = dataset
    try:
        family = normalize_family_name(dataset)
    except DatasetCaseError:
        family = dataset
    return parse_case(
        {
            "example_id": stable_example_id(family, split, index, example_id),
            "dataset": family,
            "split": split,
            "prompt": record.prompt,
            "task": resolved_task,
            "expected": record.reference,
            "choices": list(record.choices) if record.choices is not None else None,
            "answer_index": record.answer_index,
            "source": source,
            "metadata": dict(metadata or {}),
        }
    )


def case_to_record(case: DatasetCase) -> DatasetRecord:
    """Drop canonical-only fields so existing runner/metrics callers still work."""
    return DatasetRecord(
        prompt=case.prompt,
        reference=case.expected if case.task is not TaskKind.CLASSIFICATION else None,
        choices=list(case.choices) if case.choices is not None else None,
        answer_index=case.answer_index,
    )


def cases_from_loaded(
    loaded: LoadedDataset,
    *,
    task: TaskKind | None = None,
) -> list[DatasetCase]:
    source = str(loaded.metadata.get("source", loaded.spec.source))
    base_meta = {key: value for key, value in loaded.metadata.items() if key != "source"}
    if loaded.spec.hf_id:
        base_meta.setdefault("hf_id", loaded.spec.hf_id)
    if loaded.spec.hf_subset:
        base_meta.setdefault("hf_subset", loaded.spec.hf_subset)
    cases = [
        record_to_case(
            record,
            dataset=loaded.spec.name,
            split=loaded.spec.split,
            index=index,
            task=task,
            source=source,
            metadata={**base_meta, "row_index": index},
        )
        for index, record in enumerate(loaded.records)
    ]
    return validate_case_batch(cases)


def _row_to_case(
    payload: Mapping[str, Any],
    *,
    spec: FamilySpec,
    split: str,
    index: int,
    path: Path,
) -> DatasetCase:
    explicit = payload.get("example_id") or payload.get("id")
    expected = payload.get("expected", payload.get("reference"))
    row_split = str(payload.get("split", split))
    extra = payload.get("metadata")
    extra_meta = dict(extra) if isinstance(extra, Mapping) else {}
    metadata = {
        **extra_meta,
        "path": str(path),
        "row_index": index,
        "hf_id": spec.hf_id,
        "hf_subset": spec.hf_subset,
    }
    return parse_case(
        {
            "example_id": resolve_example_id(
                spec.name,
                row_split,
                index,
                str(explicit) if explicit else None,
            ),
            "dataset": spec.name,
            "split": row_split,
            "prompt": payload["prompt"],
            "task": spec.task,
            "expected": expected,
            "choices": payload.get("choices"),
            "answer_index": payload.get("answer_index"),
            "source": "jsonl",
            "metadata": metadata,
        }
    )


def _parse_jsonl_object(
    line: str,
    *,
    line_number: int,
    path: Path,
    dataset: str,
) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except ValueError as exc:
        raise DatasetCaseError(
            f"invalid json on line {line_number} in {path}",
            dataset=dataset,
            example_id=None,
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetCaseError(
            f"jsonl row must be an object (line {line_number} in {path})",
            dataset=dataset,
            example_id=None,
            expected="object",
            observed=type(payload).__name__,
        )
    if "prompt" not in payload:
        raise DatasetCaseError(
            f"missing prompt (line {line_number} in {path})",
            dataset=dataset,
            example_id=payload.get("example_id") or payload.get("id"),
            expected="prompt",
            observed=None,
        )
    return payload


def load_sample_cases(
    family: str,
    *,
    split: str = "validation",
    path: str | Path | None = None,
    max_samples: int | None = None,
) -> list[DatasetCase]:
    """Load committed local fixtures. Never downloads and never opens the network."""
    spec = family_spec(family)
    jsonl_path = Path(path) if path is not None else REPO_ROOT / spec.sample_relpath
    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset sample file not found: {jsonl_path}")
    if max_samples is not None and max_samples < 0:
        raise DatasetCaseError(
            "max_samples must be non-negative",
            dataset=spec.name,
            example_id=None,
            expected=">= 0",
            observed=max_samples,
        )

    with jsonl_path.open("r", encoding="utf-8") as handle:
        cases = _read_sample_rows(
            handle,
            spec=spec,
            split=split,
            path=jsonl_path,
            max_samples=max_samples,
        )
    return validate_case_batch(cases)


def _read_sample_rows(
    handle: Iterable[str],
    *,
    spec: FamilySpec,
    split: str,
    path: Path,
    max_samples: int | None,
) -> list[DatasetCase]:
    cases: list[DatasetCase] = []
    for line_number, line in enumerate(handle, start=1):
        if max_samples is not None and len(cases) >= max_samples:
            break
        payload = _parse_jsonl_object(
            line,
            line_number=line_number,
            path=path,
            dataset=spec.name,
        )
        if payload is None:
            continue
        cases.append(
            _row_to_case(
                payload,
                spec=spec,
                split=split,
                index=len(cases),
                path=path,
            )
        )
    return cases


def load_family_records(
    family: str,
    *,
    split: str = "validation",
    path: str | Path | None = None,
    max_samples: int | None = None,
) -> LoadedDataset:
    """Load the same local fixture through the existing JSONL loader."""
    spec = family_spec(family)
    jsonl_path = Path(path) if path is not None else REPO_ROOT / spec.sample_relpath
    dataset_spec = DatasetSpec(
        name=spec.name,
        split=split,
        source="jsonl",
        path=str(jsonl_path),
        hf_id=spec.hf_id,
        hf_subset=spec.hf_subset,
        max_samples=max_samples,
    )
    return JsonlDatasetLoader().load(dataset_spec)
