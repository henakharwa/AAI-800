"""
LCLDD Trainer — runs all 7 curriculum stages with metrics and plotting.

Stage config controls which losses are active:
  A: L_ans only
  B: L_ans (ThinkingBlock active)
  C: L_ans + L_state
  D: L_ans + L_state + L_vf
  E: L_ans + L_state + L_vf + L_lya
  F: L_ans + L_state + L_vf + L_lya + L_jac
  G: All losses + train halting head

Each stage: tracks losses, evaluates accuracy, plots curves, saves checkpoint.
"""

import os
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from ..student.lcldd_student import LCLDDStudent
from .losses import LCLDDLoss
from .metrics import MetricsTracker
from .plotter import TrainingPlotter
from .evaluator import Evaluator


# ── Stage configuration ───────────────────────────────────────────────────────

STAGE_CONFIGS = {
    "A": dict(use_lya=False, use_vf=False, use_state=False, use_jac=False,
              thinking_active=False, use_halting=False,
              description="Baseline — L_ans only"),
    "B": dict(use_lya=False, use_vf=False, use_state=False, use_jac=False,
              thinking_active=True, use_halting=False,
              description="Recursive Reasoning — L_ans + ThinkingBlock"),
    "C": dict(use_lya=False, use_vf=False, use_state=True,  use_jac=False,
              thinking_active=True, use_halting=False,
              description="State Distillation — + L_state"),
    "D": dict(use_lya=False, use_vf=True,  use_state=True,  use_jac=False,
              thinking_active=True, use_halting=False,
              description="Vector Field — + L_vf"),
    "E": dict(use_lya=True,  use_vf=True,  use_state=True,  use_jac=False,
              thinking_active=True, use_halting=False,
              description="Lyapunov Stability — + L_lya"),
    "F": dict(use_lya=True,  use_vf=True,  use_state=True,  use_jac=True,
              thinking_active=True, use_halting=False,
              description="Jacobian Alignment — + L_jac"),
    "G": dict(use_lya=True,  use_vf=True,  use_state=True,  use_jac=True,
              thinking_active=True, use_halting=True,
              description="Dynamic Halting — train halting head"),
}


class Trainer:
    """
    Runs one curriculum stage at a time.

    Usage:
        trainer = Trainer(student, tokenizer, dataloader, records, device)
        trainer.run_stage("A", max_steps=500, eval_every=100)
    """

    def __init__(
        self,
        student:    LCLDDStudent,
        tokenizer:  PreTrainedTokenizer,
        dataloader: DataLoader,
        all_records: list,
        device:     str = "cuda",
        lr:         float = 2e-5,
        grad_accum: int   = 4,
        output_dir: str   = "outputs",
        lambda_vf:    float = 0.1,
        lambda_lya:   float = 0.1,
        lambda_jac:   float = 0.05,
        lambda_state: float = 0.1,
    ):
        self.student     = student
        self.tokenizer   = tokenizer
        self.dataloader  = dataloader
        self.all_records = all_records
        self.device      = device
        self.grad_accum  = grad_accum
        self.output_dir  = Path(output_dir)

        self.loss_fn   = LCLDDLoss(lambda_vf=lambda_vf,
                                    lambda_lya=lambda_lya,
                                    lambda_jac=lambda_jac,
                                    lambda_state=lambda_state)
        self.evaluator = Evaluator(student, tokenizer, device)

        # Optimizer covers ALL student parameters
        self.optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01,
        )

        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    # ── Student Jacobian (for L_jac) ──────────────────────────────────────

    def _compute_student_jacobian(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Efficient VJP: gradient of sum(h_T) w.r.t. input embeddings."""
        embed = self.student.base_model.get_input_embeddings()
        with torch.no_grad():
            embeds = embed(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        base_out = self.student.base_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h_T  = base_out.hidden_states[-1][:, -1, :]   # (batch, hidden)
        grad = torch.autograd.grad(
            h_T.sum(), embeds, create_graph=False
        )[0]                                           # (batch, seq, hidden)
        return grad.detach()

    # ── Single training step ──────────────────────────────────────────────

    def _step(self, batch: dict, cfg: dict, compute_jac: bool) -> dict:
        """Forward + loss for one batch. Returns loss dict."""
        ids   = batch["input_ids"].to(self.device)
        mask  = batch["attention_mask"].to(self.device)
        lbls  = batch["labels"].to(self.device)
        t_hid = batch["teacher_hidden"].to(self.device)
        t_mot = batch["teacher_motion"].to(self.device)
        t_jac = batch["teacher_jacobian"].to(self.device)

        # Student forward
        out = self.student(ids, mask, use_halting=cfg["use_halting"])

        # Compute student Jacobian only when L_jac is active (expensive)
        s_jac = None
        if cfg["use_jac"] and compute_jac:
            s_jac = self._compute_student_jacobian(ids, mask)

        # h0 anchor from first hidden state
        h0 = out.hidden_states[0].to(self.device)

        loss_out = self.loss_fn(
            logits=out.logits,
            labels=lbls,
            hidden_states=[h.to(self.device) for h in out.hidden_states],
            h0_anchor=h0,
            teacher_motion=t_mot,
            teacher_hidden=t_hid,
            student_jacobian=s_jac,
            teacher_jacobian=t_jac if s_jac is not None else None,
            use_lya=cfg["use_lya"],
            use_vf=cfg["use_vf"],
            use_state=cfg["use_state"],
            use_jac=cfg["use_jac"] and s_jac is not None,
        )

        # Compute auxiliary metrics
        hs = torch.stack(out.hidden_states, dim=0)        # (T+1, B, d)
        motions = (hs[1:] - hs[:-1]).norm(dim=-1).mean().item()   # drift
        energy  = torch.stack(out.lyapunov_energies).mean().item() # energy
        steps   = out.steps_taken.float().mean().item()

        return {
            "loss_out":  loss_out,
            "drift":     motions,
            "energy":    energy,
            "steps":     steps,
        }

    # ── Run one curriculum stage ──────────────────────────────────────────

    def run_stage(
        self,
        stage:       str,
        max_steps:   int  = 1000,
        eval_every:  int  = 200,
        save_every:  int  = 500,
        jac_every:   int  = 10,    # compute Jacobian every N steps (expensive)
    ):
        cfg = STAGE_CONFIGS[stage]
        print(f"\n{'='*60}")
        print(f"  STAGE {stage}: {cfg['description']}")
        print(f"  max_steps={max_steps} | eval_every={eval_every}")
        print(f"{'='*60}")

        tracker = MetricsTracker(stage, save_dir=str(self.output_dir / "metrics"))
        self.student.train()
        self.optimizer.zero_grad()

        data_iter  = iter(self.dataloader)
        global_step = 0

        while global_step < max_steps:
            # Refill iterator when exhausted
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch     = next(data_iter)

            compute_jac = cfg["use_jac"] and (global_step % jac_every == 0)

            result   = self._step(batch, cfg, compute_jac)
            loss_out = result["loss_out"]

            # Scale for gradient accumulation
            (loss_out.total / self.grad_accum).backward()

            if (global_step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # ── Log metrics ───────────────────────────────────────────
            tracker.log_step(
                step=global_step + 1,
                losses=loss_out.to_dict(),
                lyapunov_energy=result["energy"],
                drift_magnitude=result["drift"],
                avg_steps_used=result["steps"],
            )

            # Print progress every 10 steps
            if (global_step + 1) % 10 == 0:
                loss_str = " | ".join(
                    f"{k}={v:.4f}" for k, v in loss_out.to_dict().items()
                )
                print(f"  Step {global_step+1:4d}/{max_steps} | {loss_str} | "
                      f"drift={result['drift']:.3f} | energy={result['energy']:.1f}")

            # ── Evaluate ──────────────────────────────────────────────
            if (global_step + 1) % eval_every == 0:
                print(f"\n  --- Evaluating Stage {stage} @ step {global_step+1} ---")
                accs = self.evaluator.evaluate(
                    self.all_records, max_samples=100,
                    use_halting=cfg["use_halting"],
                )
                tracker.log_eval(global_step + 1, accs)
                self.student.train()

            # ── Save checkpoint ───────────────────────────────────────
            if (global_step + 1) % save_every == 0:
                ckpt = self.output_dir / "checkpoints" / f"stage_{stage}_step{global_step+1}.pt"
                torch.save({
                    "stage":       stage,
                    "step":        global_step + 1,
                    "encoder":     self.student.encoder.state_dict(),
                    "thinking":    self.student.thinking.state_dict(),
                    "decoder":     self.student.decoder.state_dict(),
                    "halting":     self.student.halting.state_dict(),
                    "optimizer":   self.optimizer.state_dict(),
                }, ckpt)
                print(f"  [Checkpoint] Saved -> {ckpt}")

            global_step += 1

        # ── End of stage: save metrics + plots ────────────────────────
        tracker.save()
        plotter = TrainingPlotter(tracker, save_dir=str(self.output_dir / "plots"))
        plotter.plot_all()

        # Final eval
        print(f"\n  --- Final Evaluation Stage {stage} ---")
        accs = self.evaluator.evaluate(
            self.all_records, max_samples=200,
            use_halting=cfg["use_halting"],
        )
        tracker.log_eval(global_step, accs)
        tracker.save()
        tracker.summary()

        return tracker
