"""
Dataset loading and normalization for LCLDD.

All four datasets are normalized into a single common schema:
    {
        "id":       str   — unique identifier
        "question": str   — the input question/problem
        "answer":   str   — the gold answer (string-normalized)
        "dataset":  str   — source dataset name
        "split":    str   — "train" or "test"
    }

This unified format is what the teacher trajectory extractor consumes.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datasets import load_dataset, concatenate_datasets, Dataset


# ── BBH subtasks used (logic/math-heavy, best for reasoning distillation) ────

BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "temporal_sequences",
    "word_sorting",
]

# MATH subsets (algebra-heavy tasks)
MATH_SUBTASKS = [
    "algebra",
    "counting_and_probability",
    "number_theory",
    "prealgebra",
    "intermediate_algebra",
]


@dataclass
class DatasetConfig:
    name: str
    hf_id: str
    hf_config: Optional[str]
    train_split: str
    test_split: str


DATASET_CONFIGS = {
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_id="openai/gsm8k",
        hf_config="main",
        train_split="train",
        test_split="test",
    ),
    "math": DatasetConfig(
        name="math",
        hf_id="EleutherAI/hendrycks_math",
        hf_config=None,        # loaded per-subtask
        train_split="train",
        test_split="test",
    ),
    "bbh": DatasetConfig(
        name="bbh",
        hf_id="lukaemon/bbh",
        hf_config=None,        # loaded per-subtask
        train_split="test",    # BBH only has test split
        test_split="test",
    ),
    "arc_challenge": DatasetConfig(
        name="arc_challenge",
        hf_id="allenai/ai2_arc",
        hf_config="ARC-Challenge",
        train_split="train",
        test_split="test",
    ),
}


# ── Normalizers — convert each dataset's raw format to unified schema ─────────

def _normalize_gsm8k(example: dict, split: str) -> dict:
    # Answer field contains "#### 42" at the end — extract the number
    raw_answer = example["answer"]
    gold = raw_answer.split("####")[-1].strip()
    return {
        "id": f"gsm8k_{split}_{hash(example['question']) & 0xFFFFFF}",
        "question": example["question"].strip(),
        "answer": gold,
        "dataset": "gsm8k",
        "split": split,
    }


def _normalize_math(example: dict, split: str, subtask: str) -> dict:
    # Extract boxed answer from solution: \boxed{42}
    solution = example["solution"]
    gold = _extract_boxed(solution)
    return {
        "id": f"math_{subtask}_{split}_{hash(example['problem']) & 0xFFFFFF}",
        "question": example["problem"].strip(),
        "answer": gold,
        "dataset": "math",
        "split": split,
    }


def _normalize_bbh(example: dict, split: str, subtask: str) -> dict:
    return {
        "id": f"bbh_{subtask}_{split}_{hash(example['input']) & 0xFFFFFF}",
        "question": example["input"].strip(),
        "answer": str(example["target"]).strip(),
        "dataset": "bbh",
        "split": split,
    }


def _normalize_arc(example: dict, split: str) -> dict:
    # Choices are stored as {"text": [...], "label": [...]}
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = example["answerKey"]

    # Build question with lettered choices appended
    choice_str = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
    full_question = f"{example['question'].strip()}\n{choice_str}"

    # Gold answer is the text of the correct choice
    gold_idx = labels.index(answer_key) if answer_key in labels else 0
    gold = texts[gold_idx]

    return {
        "id": f"arc_{split}_{example['id']}",
        "question": full_question,
        "answer": gold,
        "dataset": "arc_challenge",
        "split": split,
    }


def _extract_boxed(text: str) -> str:
    """Extract content from LaTeX \\boxed{...}."""
    start = text.rfind("\\boxed{")
    if start == -1:
        # Fallback: last line
        return text.strip().split("\n")[-1].strip()
    depth = 0
    for i, ch in enumerate(text[start + 7:], start=start + 7):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start + 7:i]
            depth -= 1
    return text[start + 7:].strip()


# ── Public loader ──────────────────────────────────────────────────────────────

def load_all_datasets(
    cache_dir: str = "./cache/huggingface",
    datasets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    Download and normalize all datasets.

    Returns:
        {
            "gsm8k":       {"train": [...], "test": [...]},
            "math":        {"train": [...], "test": [...]},
            "bbh":         {"train": [...], "test": [...]},
            "arc_challenge":{"train": [...], "test": [...]},
        }
    """
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    result = {}

    for name in datasets:
        print(f"[Datasets] Loading {name}...")
        cfg = DATASET_CONFIGS[name]
        train_samples, test_samples = [], []

        if name == "gsm8k":
            raw_train = load_dataset(cfg.hf_id, cfg.hf_config, split=cfg.train_split, cache_dir=cache_dir)
            raw_test  = load_dataset(cfg.hf_id, cfg.hf_config, split=cfg.test_split,  cache_dir=cache_dir)
            train_samples = [_normalize_gsm8k(ex, "train") for ex in raw_train]
            test_samples  = [_normalize_gsm8k(ex, "test")  for ex in raw_test]

        elif name == "math":
            for subtask in MATH_SUBTASKS:
                try:
                    raw_train = load_dataset(cfg.hf_id, subtask, split="train", cache_dir=cache_dir)
                    raw_test  = load_dataset(cfg.hf_id, subtask, split="test",  cache_dir=cache_dir)
                    train_samples += [_normalize_math(ex, "train", subtask) for ex in raw_train]
                    test_samples  += [_normalize_math(ex, "test",  subtask) for ex in raw_test]
                except Exception as e:
                    print(f"  [Warning] Could not load MATH/{subtask}: {e}")

        elif name == "bbh":
            for subtask in BBH_SUBTASKS:
                try:
                    raw = load_dataset(cfg.hf_id, subtask, split="test", cache_dir=cache_dir)
                    samples = [_normalize_bbh(ex, "test", subtask) for ex in raw]
                    # BBH has no train split — use 80% for train, 20% for test
                    split_idx = int(len(samples) * 0.8)
                    train_samples += samples[:split_idx]
                    test_samples  += samples[split_idx:]
                except Exception as e:
                    print(f"  [Warning] Could not load BBH/{subtask}: {e}")

        elif name == "arc_challenge":
            raw_train = load_dataset(cfg.hf_id, cfg.hf_config, split=cfg.train_split, cache_dir=cache_dir)
            raw_test  = load_dataset(cfg.hf_id, cfg.hf_config, split=cfg.test_split,  cache_dir=cache_dir)
            train_samples = [_normalize_arc(ex, "train") for ex in raw_train]
            test_samples  = [_normalize_arc(ex, "test")  for ex in raw_test]

        result[name] = {"train": train_samples, "test": test_samples}
        print(f"  train: {len(train_samples):,}  |  test: {len(test_samples):,}")

    return result
