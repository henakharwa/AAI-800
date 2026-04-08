"""
FullEvaluator — comprehensive evaluation of LCLDD student model.

Metrics computed per dataset (GSM8K, MATH, BBH, ARC-Challenge):
  - Exact-match accuracy
  - Token-level F1 (partial credit)
  - Mean confidence (kappa) at final step
  - Mean entropy (eta) at final step
  - Mean steps used (halting efficiency)
  - Mean Lyapunov energy trajectory
  - Mean drift magnitude per step
  - Per-sample latency (ms)

All results saved to:
  outputs/eval/full_eval_stage_X.json
  outputs/eval/full_eval_stage_X.csv
  outputs/eval/full_eval_summary.json   (cumulative across stages)
"""

import re
import csv
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer

from ..student.lcldd_student import LCLDDStudent


# ── Answer normalisation ──────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[,\$%]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\.0+$", "", s)
    return s


def _exact_match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    return g in p


def _token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1: overlap between predicted and gold token sets.
    Gives partial credit for partially correct answers.
    """
    pred_toks = _norm(pred).split()
    gold_toks = _norm(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall    = len(common) / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ── Full Evaluator ────────────────────────────────────────────────────────────

class FullEvaluator:
    """
    Runs comprehensive evaluation on all datasets, reporting:
      - Accuracy, F1, confidence, entropy, steps, energy, drift, latency.

    Usage:
        evaluator = FullEvaluator(student, tokenizer, device, save_dir="outputs/eval")
        results = evaluator.evaluate(records, stage="A", max_samples=500,
                                     use_halting=False)
        evaluator.save(results, stage="A")
        evaluator.print_report(results)
    """

    def __init__(
        self,
        student:    LCLDDStudent,
        tokenizer:  PreTrainedTokenizer,
        device:     str = "cuda",
        save_dir:   str = "outputs/eval",
        max_new_tokens: int = 64,
    ):
        self.student   = student
        self.tokenizer = tokenizer
        self.device    = device
        self.save_dir  = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def evaluate(
        self,
        records: List[dict],
        stage: str = "A",
        max_samples: int = 500,
        use_halting: bool = False,
    ) -> Dict[str, dict]:
        """
        Evaluate on all datasets in records.

        Returns dict keyed by dataset name:
          {
            "accuracy":    float,
            "f1":          float,
            "confidence":  float,    # mean kappa at final step
            "entropy":     float,    # mean eta at final step
            "steps":       float,    # mean steps used
            "energy_traj": list,     # mean Lyapunov energy at each step
            "drift_traj":  list,     # mean drift ||h_{t+1}-h_t|| per step
            "latency_ms":  float,    # mean per-sample latency
            "n_samples":   int,
          }
        """
        self.student.eval()

        # Group by dataset
        by_dataset: Dict[str, List[dict]] = {}
        for r in records:
            ds = r.get("dataset", "unknown")
            by_dataset.setdefault(ds, []).append(r)

        all_results = {}

        for dataset, ds_records in by_dataset.items():
            samples = ds_records[:max_samples]
            print(f"  Evaluating {dataset} ({len(samples)} samples)...")

            exact_matches  = []
            f1_scores      = []
            confidences    = []
            entropies      = []
            steps_list     = []
            latencies      = []
            energy_trajs   = []   # list of lists
            drift_trajs    = []   # list of lists

            for r in samples:
                question = r["question"]
                gold     = r["gold_answer"]

                # Tokenize
                enc = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True,
                ).to(self.device)

                # Forward with timing
                t0  = time.perf_counter()
                out = self.student(
                    enc["input_ids"],
                    enc["attention_mask"],
                    use_halting=use_halting,
                )
                t1  = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)   # ms

                # Predicted answer token
                pred_token = out.logits.argmax(dim=-1)
                pred_text  = self.tokenizer.decode(pred_token, skip_special_tokens=True)

                exact_matches.append(float(_exact_match(pred_text.strip(), gold)))
                f1_scores.append(_token_f1(pred_text.strip(), gold))

                # Confidence and entropy at final hidden state
                h_final = out.hidden_states[-1]   # (1, hidden_dim)
                kappa, eta = self.student.decoder.confidence_and_entropy(h_final)
                confidences.append(kappa.mean().item())
                entropies.append(eta.mean().item())

                steps_list.append(out.steps_taken.float().mean().item())

                # Energy trajectory: shape (#steps+1,) — one value per step
                energies = [e.mean().item() for e in out.lyapunov_energies]
                energy_trajs.append(energies)

                # Drift trajectory: ||h_{t+1}-h_t|| for each consecutive pair
                hs     = out.hidden_states
                drifts = [
                    (hs[t+1] - hs[t]).norm(dim=-1).mean().item()
                    for t in range(len(hs) - 1)
                ]
                drift_trajs.append(drifts)

            # Aggregate energy/drift to mean per step position
            max_steps_seen = max(len(e) for e in energy_trajs)
            mean_energy = []
            mean_drift  = []
            for step_idx in range(max_steps_seen):
                vals = [e[step_idx] for e in energy_trajs if step_idx < len(e)]
                mean_energy.append(float(np.mean(vals)))
            for step_idx in range(max_steps_seen - 1):
                vals = [d[step_idx] for d in drift_trajs if step_idx < len(d)]
                mean_drift.append(float(np.mean(vals)))

            all_results[dataset] = {
                "accuracy":    float(np.mean(exact_matches)),
                "f1":          float(np.mean(f1_scores)),
                "confidence":  float(np.mean(confidences)),
                "entropy":     float(np.mean(entropies)),
                "steps":       float(np.mean(steps_list)),
                "energy_traj": mean_energy,
                "drift_traj":  mean_drift,
                "latency_ms":  float(np.mean(latencies)),
                "n_samples":   len(samples),
            }

            acc = all_results[dataset]["accuracy"]
            f1  = all_results[dataset]["f1"]
            print(f"    -> acc={acc:.3f}  f1={f1:.3f}  steps={all_results[dataset]['steps']:.1f}"
                  f"  latency={all_results[dataset]['latency_ms']:.1f}ms")

        self.student.train()
        return all_results

    # ── Save / Report ─────────────────────────────────────────────────────

    def save(self, results: Dict[str, dict], stage: str):
        """
        Save full evaluation results:
          1. full_eval_stage_X.json      — complete per-dataset metrics
          2. full_eval_stage_X.csv       — flat summary table
          3. full_eval_summary.json      — cumulative across stages
        """
        # 1. Per-stage JSON
        json_path = self.save_dir / f"full_eval_stage_{stage}.json"
        payload = {"stage": stage, "datasets": results}
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [Eval] Saved -> {json_path}")

        # 2. Per-stage CSV (flat summary, no trajectory lists)
        csv_path = self.save_dir / f"full_eval_stage_{stage}.csv"
        fields = ["dataset", "accuracy", "f1", "confidence", "entropy",
                  "steps", "latency_ms", "n_samples"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for ds, m in results.items():
                writer.writerow({
                    "dataset":    ds,
                    "accuracy":   round(m["accuracy"],  4),
                    "f1":         round(m["f1"],         4),
                    "confidence": round(m["confidence"], 4),
                    "entropy":    round(m["entropy"],    4),
                    "steps":      round(m["steps"],      2),
                    "latency_ms": round(m["latency_ms"], 2),
                    "n_samples":  m["n_samples"],
                })
        print(f"  [Eval] Saved -> {csv_path}")

        # 3. Cumulative summary JSON
        summary_path = self.save_dir / "full_eval_summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        # Store only the flat metrics (no trajectories) in summary
        summary[f"stage_{stage}"] = {
            ds: {k: v for k, v in m.items()
                 if k not in ("energy_traj", "drift_traj")}
            for ds, m in results.items()
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [Eval] Saved -> {summary_path}")

    def print_report(self, results: Dict[str, dict], stage: str = "?"):
        """Print a formatted summary table."""
        print(f"\n{'='*70}")
        print(f"  Full Evaluation Report — Stage {stage}")
        print(f"{'='*70}")
        header = f"  {'Dataset':<16} {'Acc':>6} {'F1':>6} {'Conf':>6} "
        header += f"{'Entr':>6} {'Steps':>6} {'Lat(ms)':>9} {'N':>5}"
        print(header)
        print("  " + "-" * 68)
        for ds, m in results.items():
            print(
                f"  {ds:<16} {m['accuracy']:>6.3f} {m['f1']:>6.3f} "
                f"{m['confidence']:>6.3f} {m['entropy']:>6.3f} "
                f"{m['steps']:>6.1f} {m['latency_ms']:>9.1f} {m['n_samples']:>5}"
            )
        print(f"{'='*70}")
