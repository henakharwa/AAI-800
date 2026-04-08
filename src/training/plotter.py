"""
TrainingPlotter — generates and saves all training curves.

Plots produced per stage:
  1. Loss curves       — total + each component (L_ans, L_lya, L_vf, L_jac)
  2. Accuracy curves   — per dataset over eval checkpoints
  3. Lyapunov energy   — mean energy across training steps
  4. Drift magnitude   — mean ||h_{t+1} - h_t|| across steps
  5. Steps used        — average recursive steps (Stage G+)
  6. Combined dashboard— all key metrics in one figure

All plots saved to outputs/plots/stage_X/
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import MetricsTracker

# Consistent colour palette
COLOURS = {
    "loss_total": "#2C3E50",
    "loss_ans":   "#E74C3C",
    "loss_lya":   "#3498DB",
    "loss_vf":    "#2ECC71",
    "loss_jac":   "#9B59B6",
    "loss_state": "#F39C12",
    "gsm8k":      "#E74C3C",
    "math":       "#3498DB",
    "bbh":        "#2ECC71",
    "arc_challenge": "#9B59B6",
}

DATASET_LABELS = {
    "gsm8k":        "GSM8K",
    "math":         "MATH",
    "bbh":          "BIG-Bench Hard",
    "arc_challenge":"ARC-Challenge",
}


class TrainingPlotter:
    """
    Generates training curves from a MetricsTracker.

    Usage:
        plotter = TrainingPlotter(tracker, save_dir="outputs/plots")
        plotter.plot_all()          # generate and save all plots
        plotter.plot_losses()       # just loss curves
        plotter.plot_accuracies()   # just accuracy curves
    """

    def __init__(
        self,
        tracker: MetricsTracker,
        save_dir: str = "outputs/plots",
    ):
        self.tracker  = tracker
        self.stage    = tracker.stage
        self.save_dir = Path(save_dir) / f"stage_{self.stage}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str):
        path = self.save_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot] Saved -> {path}")

    # ── 1. Loss curves ────────────────────────────────────────────────────

    def plot_losses(self, smooth_window: int = 50):
        losses = self.tracker.losses
        if not losses:
            return

        active = {k: v for k, v in losses.items() if v}
        n = len(active)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Stage {self.stage} — Loss Curves", fontsize=14, fontweight="bold")

        for ax, (key, vals) in zip(axes, active.items()):
            steps = self.tracker.steps[:len(vals)]
            colour = COLOURS.get(key, "#666666")

            # Raw (faint)
            ax.plot(steps, vals, alpha=0.25, color=colour, linewidth=0.8)
            # Smoothed
            smoothed = self.tracker.smoothed(key, window=smooth_window)
            ax.plot(steps, smoothed, color=colour, linewidth=2,
                    label=f"{key} (smoothed)")

            ax.set_title(key.replace("loss_", "L_"), fontsize=11)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "loss_curves")

    # ── 2. Accuracy curves ────────────────────────────────────────────────

    def plot_accuracies(self):
        accs = self.tracker.eval_accuracies
        steps = self.tracker.eval_steps
        if not accs or not steps:
            print(f"  [Plot] No eval data yet for stage {self.stage}")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Stage {self.stage} — Accuracy per Dataset",
                     fontsize=14, fontweight="bold")

        for dataset, values in accs.items():
            if not values:
                continue
            colour = COLOURS.get(dataset, "#666666")
            label  = DATASET_LABELS.get(dataset, dataset)
            ax.plot(steps[:len(values)], values, marker="o", markersize=4,
                    color=colour, linewidth=2, label=label)
            # Annotate best
            best_val = max(values)
            best_step = steps[values.index(best_val)]
            ax.annotate(f"{best_val:.3f}",
                        xy=(best_step, best_val),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=colour)

        ax.set_xlabel("Step")
        ax.set_ylabel("Exact-Match Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "accuracy_curves")

    # ── 3. Lyapunov energy ────────────────────────────────────────────────

    def plot_lyapunov_energy(self):
        energies = self.tracker.lyapunov_energy
        if not energies:
            return

        steps = self.tracker.steps[:len(energies)]
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(f"Stage {self.stage} — Lyapunov Energy V(h; x)",
                     fontsize=14, fontweight="bold")

        ax.plot(steps, energies, alpha=0.3, color=COLOURS["loss_lya"], linewidth=0.8)
        # Smoothed
        window = max(1, len(energies) // 20)
        smoothed = [
            sum(energies[max(0, i-window):i+1]) / min(i+1, window)
            for i in range(len(energies))
        ]
        ax.plot(steps, smoothed, color=COLOURS["loss_lya"], linewidth=2, label="Energy (smoothed)")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Lyapunov Energy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "lyapunov_energy")

    # ── 4. Drift magnitude ────────────────────────────────────────────────

    def plot_drift(self):
        drifts = self.tracker.drift_magnitude
        if not drifts:
            return

        steps = self.tracker.steps[:len(drifts)]
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(f"Stage {self.stage} — Drift Magnitude ||h_{{t+1}} - h_t||",
                     fontsize=14, fontweight="bold")

        ax.plot(steps, drifts, alpha=0.3, color=COLOURS["loss_vf"], linewidth=0.8)
        window = max(1, len(drifts) // 20)
        smoothed = [
            sum(drifts[max(0, i-window):i+1]) / min(i+1, window)
            for i in range(len(drifts))
        ]
        ax.plot(steps, smoothed, color=COLOURS["loss_vf"], linewidth=2, label="Drift (smoothed)")

        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Drift Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "drift_magnitude")

    # ── 5. Steps used (halting efficiency) ───────────────────────────────

    def plot_steps_used(self):
        steps_used = self.tracker.avg_steps_used
        if not steps_used:
            return

        steps = self.tracker.steps[:len(steps_used)]
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(f"Stage {self.stage} — Average Recursive Steps Used",
                     fontsize=14, fontweight="bold")

        ax.plot(steps, steps_used, color="#E67E22", linewidth=2)
        ax.axhline(max(steps_used), color="gray", linestyle="--",
                   linewidth=0.8, label=f"Max: {max(steps_used):.1f}")
        ax.axhline(min(steps_used), color="gray", linestyle=":",
                   linewidth=0.8, label=f"Min: {min(steps_used):.1f}")

        ax.set_xlabel("Step")
        ax.set_ylabel("Avg Steps Used")
        ax.set_ylim(0, None)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "steps_used")

    # ── 6. Combined dashboard ─────────────────────────────────────────────

    def plot_dashboard(self):
        """Single-figure summary of all key metrics for this stage."""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"LCLDD Training Dashboard — Stage {self.stage}",
                     fontsize=16, fontweight="bold")

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Panel 1: Total loss
        ax1 = fig.add_subplot(gs[0, 0])
        if self.tracker.losses.get("loss_total"):
            vals = self.tracker.losses["loss_total"]
            steps = self.tracker.steps[:len(vals)]
            ax1.plot(steps, vals, alpha=0.25, color=COLOURS["loss_total"])
            ax1.plot(steps, self.tracker.smoothed("loss_total"), color=COLOURS["loss_total"], linewidth=2)
        ax1.set_title("Total Loss"); ax1.set_xlabel("Step"); ax1.grid(True, alpha=0.3)

        # Panel 2: Component losses
        ax2 = fig.add_subplot(gs[0, 1])
        for key in ["loss_ans", "loss_lya", "loss_vf", "loss_jac"]:
            vals = self.tracker.losses.get(key)
            if vals:
                steps = self.tracker.steps[:len(vals)]
                ax2.plot(steps, self.tracker.smoothed(key),
                         color=COLOURS.get(key, "#666"), linewidth=1.5,
                         label=key.replace("loss_", "L_"))
        ax2.set_title("Loss Components"); ax2.set_xlabel("Step")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

        # Panel 3: Accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        for ds, accs in self.tracker.eval_accuracies.items():
            if accs:
                ax3.plot(self.tracker.eval_steps[:len(accs)], accs,
                         marker="o", markersize=3,
                         color=COLOURS.get(ds, "#666"),
                         label=DATASET_LABELS.get(ds, ds), linewidth=1.5)
        ax3.set_title("Accuracy"); ax3.set_xlabel("Step")
        ax3.set_ylim(0, 1); ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

        # Panel 4: Lyapunov energy
        ax4 = fig.add_subplot(gs[1, 0])
        if self.tracker.lyapunov_energy:
            e = self.tracker.lyapunov_energy
            ax4.plot(self.tracker.steps[:len(e)], e, color=COLOURS["loss_lya"], linewidth=1.5)
        ax4.set_title("Lyapunov Energy"); ax4.set_xlabel("Step"); ax4.grid(True, alpha=0.3)

        # Panel 5: Drift magnitude
        ax5 = fig.add_subplot(gs[1, 1])
        if self.tracker.drift_magnitude:
            d = self.tracker.drift_magnitude
            ax5.plot(self.tracker.steps[:len(d)], d, color=COLOURS["loss_vf"], linewidth=1.5)
        ax5.set_title("Drift Magnitude"); ax5.set_xlabel("Step"); ax5.grid(True, alpha=0.3)

        # Panel 6: Steps used
        ax6 = fig.add_subplot(gs[1, 2])
        if self.tracker.avg_steps_used:
            s = self.tracker.avg_steps_used
            ax6.plot(self.tracker.steps[:len(s)], s, color="#E67E22", linewidth=1.5)
        ax6.set_title("Avg Steps Used"); ax6.set_xlabel("Step"); ax6.grid(True, alpha=0.3)

        self._save(fig, "dashboard")

    # ── Plot all ──────────────────────────────────────────────────────────

    def plot_all(self):
        """Generate and save every plot for this stage."""
        print(f"\n[Plotter] Generating plots for Stage {self.stage}...")
        self.plot_losses()
        self.plot_accuracies()
        self.plot_lyapunov_energy()
        self.plot_drift()
        self.plot_steps_used()
        self.plot_dashboard()
        print(f"[Plotter] All plots saved to {self.save_dir}/")
