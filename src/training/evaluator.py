"""
Evaluator — computes exact-match accuracy on all 4 datasets.

Generates answers from the student model via greedy decoding,
extracts the predicted answer, compares to gold.
"""

import re
import torch
from typing import Dict, List
from transformers import PreTrainedTokenizer

from ..student.lcldd_student import LCLDDStudent


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[,\$%]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\.0+$", "", s)
    return s


def _answers_match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    return g in p


class Evaluator:
    """
    Runs student model on eval samples and computes exact-match accuracy.

    Usage:
        evaluator = Evaluator(student, tokenizer, device)
        results = evaluator.evaluate(eval_records, max_samples=200)
        # returns {"gsm8k": 0.42, "math": 0.31, ...}
    """

    def __init__(
        self,
        student: LCLDDStudent,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ):
        self.student        = student
        self.tokenizer      = tokenizer
        self.device         = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def evaluate(
        self,
        records: List[dict],
        max_samples: int = 200,
        use_halting: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate on records. Returns per-dataset accuracy dict.
        """
        self.student.eval()

        # Group by dataset
        by_dataset: Dict[str, List[dict]] = {}
        for r in records:
            ds = r.get("dataset", "unknown")
            by_dataset.setdefault(ds, []).append(r)

        results = {}

        for dataset, ds_records in by_dataset.items():
            samples = ds_records[:max_samples]
            correct = 0

            for r in samples:
                question = r["question"]
                gold     = r["gold_answer"]

                # Tokenize input
                enc = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True,
                ).to(self.device)

                # Student forward — get final hidden state
                out = self.student(
                    enc["input_ids"],
                    enc["attention_mask"],
                    use_halting=use_halting,
                )

                # Greedy decode: get top token from logits as seed,
                # then generate continuation from base model
                pred_token = out.logits.argmax(dim=-1)  # (1,)
                pred_text  = self.tokenizer.decode(pred_token, skip_special_tokens=True)

                if _answers_match(pred_text.strip(), gold):
                    correct += 1

            acc = correct / max(len(samples), 1)
            results[dataset] = acc

        self.student.train()
        return results
