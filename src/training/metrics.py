"""
MetricsTracker — records all training and evaluation metrics across stages.

Tracks per-step:
  - All loss components (total, ans, lya, vf, jac, state)
  - Accuracy per dataset (GSM8K, MATH, BBH, ARC-Challenge)
  - Lyapunov energy (mean across batch and steps)
  - Drift magnitude (mean ||h_{t+1} - h_t||)
  - Average steps used (for halting efficiency)

At each eval checkpoint, a full performance snapshot is saved containing
accuracies together with all other metrics (losses, energy, drift, steps).

Files written per stage:
  stage_X_metrics.json           — full step-level training history
  stage_X_performance.json       — eval-checkpoint snapshots (accs + metrics)
  stage_X_performance.csv        — same data in tabular form
  all_stages_performance.json    — cumulative across all stages
"""

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class MetricsTracker:
    """
    Records metrics during training. One tracker per curriculum stage.

    Usage:
        tracker = MetricsTracker("E", save_dir="outputs/metrics")
        tracker.log_step(step=1, losses={"loss_total": 1.2, "loss_lya": 0.3}, ...)
        tracker.log_eval(step=100, accuracies={"gsm8k": 0.42, "math": 0.31})
        tracker.save()
    """

    def __init__(self, stage: str, save_dir: str = "outputs/metrics"):
        self.stage    = stage
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Step-level training history
        self.steps:            List[int]   = []
        self.losses:           Dict[str, List[float]] = defaultdict(list)
        self.lyapunov_energy:  List[float] = []
        self.drift_magnitude:  List[float] = []
        self.avg_steps_used:   List[float] = []
        self.timestamps:       List[float] = []

        # Eval checkpoints (less frequent)
        self.eval_steps:       List[int]   = []
        self.eval_accuracies:  Dict[str, List[float]] = defaultdict(list)

        # Snapshot of all metrics captured at each eval checkpoint
        # Each entry mirrors exactly one element of eval_steps
        self.eval_losses:       Dict[str, List[float]] = defaultdict(list)
        self.eval_lyapunov:     List[float] = []
        self.eval_drift:        List[float] = []
        self.eval_steps_used:   List[float] = []

        self._start_time = time.time()

    # ── Training step logging ─────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        losses: Dict[str, float],
        lyapunov_energy: Optional[float] = None,
        drift_magnitude: Optional[float] = None,
        avg_steps_used:  Optional[float] = None,
    ):
        """Call once per training step."""
        self.steps.append(step)
        self.timestamps.append(time.time() - self._start_time)

        for k, v in losses.items():
            self.losses[k].append(v)

        if lyapunov_energy is not None:
            self.lyapunov_energy.append(lyapunov_energy)
        if drift_magnitude is not None:
            self.drift_magnitude.append(drift_magnitude)
        if avg_steps_used is not None:
            self.avg_steps_used.append(avg_steps_used)

    # ── Evaluation logging ────────────────────────────────────────────────

    def log_eval(self, step: int, accuracies: Dict[str, float]):
        """
        Call at evaluation checkpoints.
        Captures a full performance snapshot: accuracies + latest losses,
        Lyapunov energy, drift, and steps used at this moment in training.
        Saves immediately to disk after every call.
        """
        self.eval_steps.append(step)
        for dataset, acc in accuracies.items():
            self.eval_accuracies[dataset].append(acc)

        # Snapshot the latest training metrics at this eval point
        latest_losses = self.latest_losses()
        for k, v in latest_losses.items():
            self.eval_losses[k].append(v)
        self.eval_lyapunov.append(
            self.lyapunov_energy[-1] if self.lyapunov_energy else 0.0)
        self.eval_drift.append(
            self.drift_magnitude[-1] if self.drift_magnitude else 0.0)
        self.eval_steps_used.append(
            self.avg_steps_used[-1] if self.avg_steps_used else 0.0)

        print(f"  [Eval @ step {step}] " +
              " | ".join(f"{k}: {v:.3f}" for k, v in accuracies.items()))

        # Save full performance snapshot immediately after every eval checkpoint
        self._save_performance()

    # ── Computed properties ───────────────────────────────────────────────

    def latest_losses(self) -> Dict[str, float]:
        """Returns the most recent value for each tracked loss."""
        return {k: v[-1] for k, v in self.losses.items() if v}

    def smoothed(self, key: str, window: int = 50) -> List[float]:
        """Moving average of a loss series."""
        vals = self.losses.get(key, [])
        if not vals:
            return []
        smoothed = []
        for i in range(len(vals)):
            start = max(0, i - window + 1)
            smoothed.append(sum(vals[start:i+1]) / (i - start + 1))
        return smoothed

    # ── Save / Load ───────────────────────────────────────────────────────

    def _save_performance(self):
        """
        Save a complete performance snapshot after every eval checkpoint.

        Writes three files:
          1. stage_X_performance.json      — accuracies + all metrics per eval step
          2. stage_X_performance.csv       — same data in tabular form (Excel-friendly)
          3. all_stages_performance.json   — cumulative across all stages (best/latest)
        """
        datasets     = list(self.eval_accuracies.keys())
        loss_keys    = list(self.eval_losses.keys())

        # 1. Per-stage JSON — full history
        perf = {
            "stage":      self.stage,
            "eval_steps": self.eval_steps,

            # Accuracies at each eval step
            "accuracies": dict(self.eval_accuracies),

            # Loss components at each eval step
            "losses_at_eval": dict(self.eval_losses),

            # Stability/efficiency metrics at each eval step
            "lyapunov_at_eval":   self.eval_lyapunov,
            "drift_at_eval":      self.eval_drift,
            "steps_used_at_eval": self.eval_steps_used,

            # Summaries
            "best_accuracy": {
                ds: max(vals)
                for ds, vals in self.eval_accuracies.items() if vals
            },
            "latest_accuracy": {
                ds: vals[-1]
                for ds, vals in self.eval_accuracies.items() if vals
            },
            "best_loss_total": (
                min(self.eval_losses["loss_total"])
                if self.eval_losses.get("loss_total") else None
            ),
            "latest_loss_total": (
                self.eval_losses["loss_total"][-1]
                if self.eval_losses.get("loss_total") else None
            ),
        }
        json_path = self.save_dir / f"stage_{self.stage}_performance.json"
        with open(json_path, "w") as f:
            json.dump(perf, f, indent=2)

        # 2. Per-stage CSV — one row per eval checkpoint, all columns together
        csv_path = self.save_dir / f"stage_{self.stage}_performance.csv"
        header = (["step"]
                  + [f"acc_{ds}" for ds in datasets]
                  + loss_keys
                  + ["lyapunov", "drift", "steps_used"])
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, step in enumerate(self.eval_steps):
                row = [step]
                # Accuracy columns
                for ds in datasets:
                    vals = self.eval_accuracies[ds]
                    row.append(round(vals[i], 4) if i < len(vals) else "")
                # Loss columns
                for k in loss_keys:
                    vals = self.eval_losses[k]
                    row.append(round(vals[i], 4) if i < len(vals) else "")
                # Stability columns
                row.append(round(self.eval_lyapunov[i], 4)
                           if i < len(self.eval_lyapunov) else "")
                row.append(round(self.eval_drift[i], 4)
                           if i < len(self.eval_drift) else "")
                row.append(round(self.eval_steps_used[i], 2)
                           if i < len(self.eval_steps_used) else "")
                writer.writerow(row)

        # 3. Cumulative all-stages JSON — best + latest per stage
        all_path = self.save_dir / "all_stages_performance.json"
        all_data = {}
        if all_path.exists():
            with open(all_path) as f:
                all_data = json.load(f)
        all_data[f"stage_{self.stage}"] = {
            "eval_steps":         self.eval_steps,
            "accuracies":         dict(self.eval_accuracies),
            "losses_at_eval":     dict(self.eval_losses),
            "lyapunov_at_eval":   self.eval_lyapunov,
            "drift_at_eval":      self.eval_drift,
            "steps_used_at_eval": self.eval_steps_used,
            "best_accuracy":      perf["best_accuracy"],
            "latest_accuracy":    perf["latest_accuracy"],
            "best_loss_total":    perf["best_loss_total"],
            "latest_loss_total":  perf["latest_loss_total"],
        }
        with open(all_path, "w") as f:
            json.dump(all_data, f, indent=2)

    def save(self):
        """Save full step-level training history and final performance snapshot."""
        # Full step-level history
        path = self.save_dir / f"stage_{self.stage}_metrics.json"
        data = {
            "stage":           self.stage,
            "steps":           self.steps,
            "losses":          dict(self.losses),
            "lyapunov_energy": self.lyapunov_energy,
            "drift_magnitude": self.drift_magnitude,
            "avg_steps_used":  self.avg_steps_used,
            "timestamps":      self.timestamps,
            "eval_steps":      self.eval_steps,
            "eval_accuracies": dict(self.eval_accuracies),
            "eval_losses":     dict(self.eval_losses),
            "eval_lyapunov":   self.eval_lyapunov,
            "eval_drift":      self.eval_drift,
            "eval_steps_used": self.eval_steps_used,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [Metrics]     Saved -> {path}")

        # Performance snapshot (accuracies + all metrics together)
        self._save_performance()
        print(f"  [Performance] Saved -> {self.save_dir / f'stage_{self.stage}_performance.json'}")
        print(f"  [Performance] Saved -> {self.save_dir / f'stage_{self.stage}_performance.csv'}")
        print(f"  [Performance] Saved -> {self.save_dir / 'all_stages_performance.json'}")

    @classmethod
    def load(cls, stage: str, save_dir: str = "outputs/metrics") -> "MetricsTracker":
        path = Path(save_dir) / f"stage_{stage}_metrics.json"
        with open(path) as f:
            data = json.load(f)
        tracker = cls(stage, save_dir)
        tracker.steps            = data["steps"]
        tracker.losses           = defaultdict(list, data["losses"])
        tracker.lyapunov_energy  = data["lyapunov_energy"]
        tracker.drift_magnitude  = data["drift_magnitude"]
        tracker.avg_steps_used   = data["avg_steps_used"]
        tracker.timestamps       = data["timestamps"]
        tracker.eval_steps       = data["eval_steps"]
        tracker.eval_accuracies  = defaultdict(list, data["eval_accuracies"])
        # New eval-snapshot fields (may not exist in old saves)
        tracker.eval_losses      = defaultdict(list, data.get("eval_losses", {}))
        tracker.eval_lyapunov    = data.get("eval_lyapunov", [])
        tracker.eval_drift       = data.get("eval_drift", [])
        tracker.eval_steps_used  = data.get("eval_steps_used", [])
        return tracker

    def summary(self):
        print(f"\n[Stage {self.stage}] Metrics summary:")
        print(f"  Steps logged : {len(self.steps)}")
        if self.steps:
            for k, v in self.latest_losses().items():
                print(f"  {k:<20}: {v:.4f} (latest)")
        if self.eval_lyapunov:
            print(f"  Lyapunov energy: latest={self.eval_lyapunov[-1]:.3f}")
        if self.eval_drift:
            print(f"  Drift magnitude: latest={self.eval_drift[-1]:.3f}")
        if self.eval_steps_used:
            print(f"  Steps used     : latest={self.eval_steps_used[-1]:.2f}")
        if self.eval_steps:
            print(f"  Eval checkpoints: {len(self.eval_steps)}")
            for ds, accs in self.eval_accuracies.items():
                print(f"    {ds:<15}: best={max(accs):.3f}  latest={accs[-1]:.3f}")
