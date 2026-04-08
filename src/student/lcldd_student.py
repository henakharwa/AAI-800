"""
LCLDDStudent — the full student model wiring Encoder + ThinkingBlock + Decoder.

Forward pass (Eq. 1 from paper):
    h_0^S  = E(x)
    h_{t+1}^S = h_t^S + G(h_t^S, x)     for t = 0 ... T-1
    y_hat  = D(h_T^S)

Also implements:
  - Lyapunov energy V(h; x)              — used by L_lya
  - Dynamic Phase-Space halting          — adaptive compute per input
  - Returns full trajectory {h_t^S}      — used by L_vf and L_jac
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .thinking_block import ThinkingBlock
from .decoder import Decoder


# ── Forward output container ──────────────────────────────────────────────────

@dataclass
class StudentOutput:
    logits: torch.Tensor            # (batch, vocab_size) — final answer logits
    hidden_states: List[torch.Tensor]  # list of (batch, hidden_dim), one per step
    steps_taken: torch.Tensor       # (batch,) int — how many steps each sample used
    lyapunov_energies: List[torch.Tensor]  # V(h_t; x) at each step


# ── Halting controller ────────────────────────────────────────────────────────

class HaltingController(nn.Module):
    """
    Monitors four convergence signals and decides when to stop recursion.

    Halts at step t when ALL four conditions hold (Eq. 9):
        delta_t < tau_delta   (latent change small)
        E_t     < tau_E       (Lyapunov energy low)
        kappa_t > tau_kappa   (answer confidence high)
        eta_t   < tau_eta     (prediction entropy low)

    Thresholds are learnable parameters trained with binary cross-entropy.
    """

    def __init__(self):
        super().__init__()
        # Initialised to permissive values — controller learns to tighten them
        self.log_tau_delta  = nn.Parameter(torch.tensor(-2.0))   # ~0.13
        self.log_tau_energy = nn.Parameter(torch.tensor(0.0))    # ~1.0
        self.log_tau_kappa  = nn.Parameter(torch.tensor(-0.1))   # ~0.90
        self.log_tau_eta    = nn.Parameter(torch.tensor(-1.2))   # ~0.30

    @property
    def tau_delta(self):  return self.log_tau_delta.exp()
    @property
    def tau_energy(self): return self.log_tau_energy.exp()
    @property
    def tau_kappa(self):  return torch.sigmoid(self.log_tau_kappa)
    @property
    def tau_eta(self):    return self.log_tau_eta.exp()

    def should_halt(
        self,
        delta_t: torch.Tensor,   # (batch,)
        energy_t: torch.Tensor,  # (batch,)
        kappa_t: torch.Tensor,   # (batch,)
        eta_t: torch.Tensor,     # (batch,)
    ) -> torch.Tensor:
        """Returns bool tensor (batch,): True = halt this sample."""
        return (
            (delta_t  < self.tau_delta)  &
            (energy_t < self.tau_energy) &
            (kappa_t  > self.tau_kappa)  &
            (eta_t    < self.tau_eta)
        )


# ── Full student model ─────────────────────────────────────────────────────────

class LCLDDStudent(nn.Module):
    """
    Full LCLDD student: Encoder + ThinkingBlock + Decoder + HaltingController.

    Built on top of a pre-trained base model (Phi-3.5-mini or Qwen2.5-1.5B).
    The base model provides the initial hidden representation of the input.
    The ThinkingBlock then iteratively refines it.
    """

    def __init__(
        self,
        base_model: nn.Module,          # pre-trained student LLM
        hidden_dim: int,                # student hidden dimension
        vocab_size: int,                # vocabulary size
        t_max: int = 20,                # max recursive steps
        alpha: float = 1.0,             # Lyapunov proximity weight
        beta: float = 0.1,              # Lyapunov norm weight
        mlp_expansion: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()

        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.t_max = t_max
        self.alpha = alpha
        self.beta = beta

        # ── LCLDD components ──────────────────────────────────────────────
        self.encoder  = Encoder(hidden_dim, hidden_dim)
        self.thinking = ThinkingBlock(hidden_dim, mlp_expansion, n_heads)
        self.decoder  = Decoder(hidden_dim, vocab_size)
        self.halting  = HaltingController()

    # ── Lyapunov energy ───────────────────────────────────────────────────

    def lyapunov_energy(
        self,
        h: torch.Tensor,    # (batch, hidden_dim) — current state
        h0: torch.Tensor,   # (batch, hidden_dim) — semantic anchor
    ) -> torch.Tensor:
        """
        V(h; x) = alpha * ||h - h0||^2 + beta * ||h||^2   (Eq. 3)
        Returns: (batch,)
        """
        proximity = (h - h0).pow(2).sum(dim=-1)
        norm_sq   = h.pow(2).sum(dim=-1)
        return self.alpha * proximity + self.beta * norm_sq

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,         # (batch, seq_len)
        attention_mask: torch.Tensor,    # (batch, seq_len)
        use_halting: bool = False,       # True only in Stage G+
    ) -> StudentOutput:
        """
        Full recursive forward pass.

        Steps:
          1. Base model encodes input -> extract last-token hidden state
          2. Encoder projects to h_0 (semantic anchor)
          3. ThinkingBlock loops for t_max steps (or until halting fires)
          4. Decoder produces final logits from h_T
        """
        batch = input_ids.shape[0]

        # ── Step 1: Base model forward — get last-token hidden state ──────
        with torch.no_grad():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Last layer, last token position per sample
        seq_lens = attention_mask.sum(dim=1) - 1          # (batch,)
        last_hidden = base_out.hidden_states[-1]           # (batch, seq, hidden_dim)
        # Gather the last real token for each sample
        idx = seq_lens.clamp(min=0).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, 1, last_hidden.shape[-1])
        x_repr = last_hidden.gather(1, idx).squeeze(1)    # (batch, hidden_dim)

        # Cast to encoder dtype (fp32) — base model runs fp16, LCLDD components fp32
        x_repr = x_repr.to(self.encoder.projection.weight.dtype)

        # ── Step 2: Encode -> h_0 (fixed semantic anchor) ─────────────────
        h0 = self.encoder(x_repr)   # (batch, hidden_dim)
        h  = h0.clone()             # h_0^S = h_0

        # ── Step 3: Recursive thinking ─────────────────────────────────────
        hidden_states    = [h.detach()]
        lyapunov_energies = [self.lyapunov_energy(h, h0).detach()]
        steps_taken      = torch.zeros(batch, dtype=torch.long, device=h.device)
        halted           = torch.zeros(batch, dtype=torch.bool,  device=h.device)

        for t in range(self.t_max):
            delta = self.thinking(h, h0)           # (batch, hidden_dim)
            h_new = h + delta

            if use_halting:
                # Compute all four halting signals
                delta_norm = delta.norm(dim=-1)                        # (batch,)
                energy     = self.lyapunov_energy(h_new, h0)          # (batch,)
                kappa, eta = self.decoder.confidence_and_entropy(h_new)

                halt_now = self.halting.should_halt(delta_norm, energy, kappa, eta)
                halt_now = halt_now & ~halted  # only first-time halt

                # Samples that halted keep their previous h; others advance
                h = torch.where(halted.unsqueeze(-1), h, h_new)
                halted = halted | halt_now
                steps_taken[~halted] = t + 1
            else:
                h = h_new
                steps_taken[:] = t + 1

            hidden_states.append(h.detach())
            lyapunov_energies.append(self.lyapunov_energy(h, h0).detach())

            if use_halting and halted.all():
                break  # all samples in batch have halted

        # ── Step 4: Decode final hidden state ─────────────────────────────
        logits = self.decoder(h)   # (batch, vocab_size)

        return StudentOutput(
            logits=logits,
            hidden_states=hidden_states,
            steps_taken=steps_taken,
            lyapunov_energies=lyapunov_energies,
        )

    def get_encoder_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns h_0 (the semantic anchor) for a given input.
        Used separately when computing the Jacobian loss.
        """
        with torch.no_grad():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        seq_lens = attention_mask.sum(dim=1) - 1
        last_hidden = base_out.hidden_states[-1]
        idx = seq_lens.clamp(min=0).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, 1, last_hidden.shape[-1])
        x_repr = last_hidden.gather(1, idx).squeeze(1)
        x_repr = x_repr.to(self.encoder.projection.weight.dtype)
        return self.encoder(x_repr)
