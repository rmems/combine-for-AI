from __future__ import annotations

from benchmarks.datasets import DatasetRecord, DatasetSpec, LoadedDataset


def load_lambada_sample(sample_size: int = 50, seed: int = 42) -> list[DatasetRecord]:
    """
    Load a small deterministic **synthetic** cloze sample for smoke tests.

    This is not the real LAMBADA corpus (see issue #11). Prompts are toy
    completion pairs with a fixed seed for reproducibility.
    """
    import random

    rng = random.Random(seed)

    examples = [
        ("She walked to the ", "store"),
        ("He sat on the ", "chair"),
        ("The cat chased the ", "mouse"),
        ("She opened the ", "door"),
        ("He drove the ", "car"),
        ("The sun shines in the ", "sky"),
        ("She wrote a ", "letter"),
        ("He played the ", "piano"),
        ("The bird flew into the ", "tree"),
        ("She drank a cup of ", "coffee"),
        ("He read the ", "book"),
        ("The dog barked at the ", "mailman"),
        ("She cooked a delicious ", "meal"),
        ("He fixed the broken ", "window"),
        ("The children played in the ", "park"),
        ("She painted a beautiful ", "picture"),
        ("He built a tall ", "tower"),
        ("The river flows through the ", "valley"),
        ("She sang a lovely ", "song"),
        ("He climbed the tall ", "mountain"),
        ("The train arrived at the ", "station"),
        ("She planted flowers in the ", "garden"),
        ("He caught a big ", "fish"),
        ("The plane flew over the ", "ocean"),
        ("She baked a chocolate ", "cake"),
        ("He repaired the old ", "bicycle"),
        ("The wind blew through the ", "trees"),
        ("She knitted a warm ", "sweater"),
        ("He solved the difficult ", "problem"),
        ("The moon shines at ", "night"),
        ("She washed the dirty ", "dishes"),
        ("He cleaned the entire ", "house"),
        ("The fire burned in the ", "fireplace"),
        ("She organized her messy ", "desk"),
        ("He painted the white ", "fence"),
        ("The rain fell from the ", "clouds"),
        ("She packed her suitcase for the ", "trip"),
        ("He fixed the leaky ", "faucet"),
        ("The snow covered the entire ", "ground"),
        ("She decorated the Christmas ", "tree"),
        ("He learned to play the ", "guitar"),
        ("The stars twinkle in the ", "sky"),
        ("She prepared a healthy ", "salad"),
        ("He installed the new ", "software"),
        ("The leaves fell from the ", "trees"),
        ("She designed a beautiful ", "dress"),
        ("He completed the challenging ", "puzzle"),
        ("The waves crashed on the ", "shore"),
        ("She taught the eager ", "students"),
        ("He won the chess ", "tournament"),
    ]
    
    # Select a deterministic sample
    rng.shuffle(examples)
    selected = examples[:sample_size]
    
    return [
        DatasetRecord(prompt=prompt, reference=reference)
        for prompt, reference in selected
    ]


class LAMBADALoader:
    """Synthetic cloze loader registered as ``lambada`` for smoke paths (not HF LAMBADA)."""

    def __init__(self, sample_size: int = 50, seed: int = 42):
        self.sample_size = sample_size
        self.seed = seed

    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if spec.max_samples is None:
            sample_size = self.sample_size
        else:
            sample_size = spec.max_samples
            if sample_size < 0:
                raise ValueError("max_samples must be non-negative")

        records = load_lambada_sample(
            sample_size=sample_size,
            seed=self.seed,
        )

        return LoadedDataset(
            spec=spec,
            records=records,
            metadata={
                "source": "synthetic_cloze_sample",
                "sample_size": len(records),
                "seed": self.seed,
            },
        )


def calculate_cloze_accuracy(
    predictions: list[str], references: list[str]
) -> float:
    """Calculate cloze accuracy; empty+empty is 0.0, length mismatch raises."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return 0.0

    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return correct / len(predictions)


def calculate_perplexity(
    log_probabilities: list[float],
    token_counts: list[int],
) -> float:
    """Calculate perplexity from token log-probs; validate counts before fallback."""
    total_tokens = sum(token_counts)
    if len(log_probabilities) != total_tokens:
        raise ValueError(
            f"Log probabilities ({len(log_probabilities)}) must match "
            f"total tokens ({total_tokens})"
        )
    if total_tokens == 0:
        return float("inf")

    avg_log_prob = sum(log_probabilities) / total_tokens
    return float(2 ** (-avg_log_prob))