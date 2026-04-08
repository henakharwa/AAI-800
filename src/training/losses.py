"""
LCLDD Loss Functions (Eq. 10 from paper):

    L = L_ans + lambda_vf * L_vf + lambda_lya * L_lya + lambda_jac * L_jac

Four components:
  L_ans  — task answer loss (cross-entropy)
  L_lya  — Lyapunov stability: enforces energy descent at every step
  L_vf   — vector field: student motion vectors match teacher motion vectors
  L_jac  — Jacobian alignment: student input sensitivity matches teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from dataclasses import dataclass


# ── Loss output container ─────────────────────────────────────────────────────

@dataclass
class LossOutput:
    total:    torch.Tensor       # combined weighted loss (backward on this)
    ans:      torch.Tensor       # L_ans
    lya:      Optional[torch.Tensor] = None   # L_lya
    vf:       Optional[torch.Tensor] = None   # L_vf
    jac:      Optional[torch.Tensor] = None   # L_jac
    state:    Optional[torch.Tensor] = None   # L_state (Stage C)

    def to_dict(self) -> Dict[str, float]:
        """Returns scalar float dict for logging."""
        d = {"loss_total": self.total.item(), "loss_ans": self.ans.item()}
        if self.lya   is not None: d["loss_lya"]   = self.lya.item()
        if self.vf    is not None: d["loss_vf"]    = self.vf.item()
        if self.jac   is not None: d["loss_jac"]   = self.jac.item()
        if self.state is not None: d["loss_state"] = self.state.item()
        return d


# ── Individual loss functions ─────────────────────────────────────────────────

def loss_ans(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    L_ans — cross-entropy answer loss.

    Args:
        logits: (batch, vocab_size)
        labels: (batch,) — gold token ids
    """
    return F.cross_entropy(logits, labels)


def loss_lyapunov(
    hidden_states: List[torch.Tensor],   # list of (batch, hidden_dim), one per step
    h0: torch.Tensor,                    # (batch, hidden_dim) — semantic anchor
    alpha: float = 1.0,
    beta: float = 0.1,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    L_lya — Lyapunov stability loss (Eq. 4):

        L_lya = sum_{t=0}^{T-1} max(0, V(h_{t+1}) - V(h_t) + epsilon)

    Penalises any step where energy does NOT decrease by at least epsilon.
    When minimised to zero, energy strictly descends -> trajectory converges.

    Args:
        hidden_states: [h_0, h_1, ..., h_T]  T+1 tensors
        h0:            fixed semantic anchor (same as hidden_states[0] before training)
        alpha, beta:   Lyapunov energy hyperparameters
        epsilon:       stability margin (strict descent requirement)
    """
    def energy(h):
        proximity = (h - h0).pow(2).sum(dim=-1)   # (batch,)
        norm_sq   = h.pow(2).sum(dim=-1)           # (batch,)
        return alpha * proximity + beta * norm_sq   # (batch,)

    total = torch.tensor(0.0, device=h0.device, requires_grad=True)
    for t in range(len(hidden_states) - 1):
        e_t   = energy(hidden_states[t])
        e_t1  = energy(hidden_states[t + 1])
        # Penalise energy increase: max(0, V(t+1) - V(t) + epsilon)
        violation = F.relu(e_t1 - e_t + epsilon)   # (batch,)
        total = total + violation.mean()

    return total


def loss_vector_field(
    student_hidden: List[torch.Tensor],  # [h_0^S .. h_T^S], each (batch, hidden_dim)
    teacher_motion: torch.Tensor,        # (batch, N-1, hidden_dim) — v_t^T from cache
) -> torch.Tensor:
    """
    L_vf — vector field distillation loss (Eq. 6):

        L_vf = sum_{t=0}^{T-1} || v_t^S - v_t^T ||^2

    where v_t^S = h_{t+1}^S - h_t^S  (student motion vector)
    and   v_t^T                       (teacher motion vector from trajectory cache)

    Enforces trajectory-level alignment, not just endpoint matching.
    """
    # Student motion vectors: v_t^S = h_{t+1} - h_t
    student_motions = []
    for t in range(len(student_hidden) - 1):
        v_s = student_hidden[t + 1] - student_hidden[t]  # (batch, hidden_dim)
        student_motions.append(v_s)

    student_motions = torch.stack(student_motions, dim=1)  # (batch, T, hidden_dim)

    # Align number of steps: use min(T_student, T_teacher)
    T = min(student_motions.shape[1], teacher_motion.shape[1])
    s = student_motions[:, :T, :]   # (batch, T, hidden_dim)
    t = teacher_motion[:, :T, :]    # (batch, T, hidden_dim)

    # Align hidden dims via mean-pooling if sizes differ
    if s.shape[-1] != t.shape[-1]:
        t = t.mean(dim=-1, keepdim=True).expand_as(s)

    return (s - t).pow(2).mean()


def loss_state(
    student_hidden: List[torch.Tensor],  # [h_0^S .. h_T^S]
    teacher_hidden: torch.Tensor,        # (batch, N, hidden_dim) — from cache
) -> torch.Tensor:
    """
    L_state — snapshot hidden-state matching loss (Stage C only).
    Simpler than L_vf: matches hidden states pointwise rather than motion vectors.
    Used before L_vf is introduced to warm up the thinking block.
    """
    T = min(len(student_hidden), teacher_hidden.shape[1])
    total = torch.tensor(0.0, device=teacher_hidden.device, requires_grad=True)
    for t in range(T):
        h_s = student_hidden[t]              # (batch, d_student)
        h_t = teacher_hidden[:, t, :]       # (batch, d_teacher)
        # Align dims
        if h_s.shape[-1] != h_t.shape[-1]:
            h_t = h_t[:, :h_s.shape[-1]] if h_t.shape[-1] > h_s.shape[-1] \
                  else F.pad(h_t, (0, h_s.shape[-1] - h_t.shape[-1]))
        total = total + (h_s - h_t).pow(2).mean()
    return total / max(T, 1)


def loss_jacobian(
    student_jacobian: torch.Tensor,   # (batch, seq_len, hidden_dim_s)
    teacher_jacobian: torch.Tensor,   # (batch, seq_len, hidden_dim_t)
) -> torch.Tensor:
    """
    L_jac — Jacobian manifold alignment loss (Eq. 8):

        L_jac = || J^S(x) - J^T(x) ||_F^2

    Aligns input sensitivity between student and teacher.
    Forces student to attend to same input tokens as teacher (causal reasoning).
    """
    # Align seq_len and hidden_dim by truncating to min
    seq  = min(student_jacobian.shape[1], teacher_jacobian.shape[1])
    dim  = min(student_jacobian.shape[2], teacher_jacobian.shape[2])

    js = student_jacobian[:, :seq, :dim]
    jt = teacher_jacobian[:, :seq, :dim]

    return (js - jt).pow(2).mean()


# ── Combined loss module ───────────────────────────────────────────────────────

class LCLDDLoss(nn.Module):
    """
    Combined LCLDD loss (Eq. 10):

        L = L_ans + lambda_vf * L_vf + lambda_lya * L_lya + lambda_jac * L_jac

    Active losses are controlled per curriculum stage.
    Pass only the tensors for losses that are active in the current stage.
    """

    def __init__(
        self,
        lambda_vf:    float = 0.1,
        lambda_lya:   float = 0.1,
        lambda_jac:   float = 0.05,
        lambda_state: float = 0.1,  # weight for L_state (was unscaled — bug fix)
        alpha:        float = 1.0,  # Lyapunov proximity weight
        beta:         float = 0.1,  # Lyapunov norm weight
        epsilon:      float = 0.01, # Lyapunov stability margin
    ):
        super().__init__()
        self.lambda_vf    = lambda_vf
        self.lambda_lya   = lambda_lya
        self.lambda_jac   = lambda_jac
        self.lambda_state = lambda_state
        self.alpha        = alpha
        self.beta         = beta
        self.epsilon      = epsilon

    def forward(
        self,
        # Always required
        logits:          torch.Tensor,
        labels:          torch.Tensor,
        # Required when active
        hidden_states:   Optional[List[torch.Tensor]] = None,  # [h0..hT]
        h0_anchor:       Optional[torch.Tensor]       = None,  # semantic anchor
        teacher_motion:  Optional[torch.Tensor]       = None,  # (B, N-1, d_t)
        teacher_hidden:  Optional[torch.Tensor]       = None,  # (B, N, d_t)
        student_jacobian:Optional[torch.Tensor]       = None,  # (B, seq, d_s)
        teacher_jacobian:Optional[torch.Tensor]       = None,  # (B, seq, d_t)
        # Stage flags
        use_lya:   bool = False,
        use_vf:    bool = False,
        use_state: bool = False,
        use_jac:   bool = False,
    ) -> LossOutput:

        # L_ans always active
        l_ans = loss_ans(logits, labels)
        total = l_ans

        l_lya = l_vf = l_state = l_jac = None

        if use_lya and hidden_states is not None and h0_anchor is not None:
            l_lya  = loss_lyapunov(hidden_states, h0_anchor, self.alpha, self.beta, self.epsilon)
            total  = total + self.lambda_lya * l_lya

        if use_vf and hidden_states is not None and teacher_motion is not None:
            l_vf  = loss_vector_field(hidden_states, teacher_motion)
            total = total + self.lambda_vf * l_vf

        if use_state and hidden_states is not None and teacher_hidden is not None:
            l_state = loss_state(hidden_states, teacher_hidden)
            total   = total + self.lambda_state * l_state

        if use_jac and student_jacobian is not None and teacher_jacobian is not None:
            l_jac = loss_jacobian(student_jacobian, teacher_jacobian)
            total = total + self.lambda_jac * l_jac

        return LossOutput(
            total=total, ans=l_ans,
            lya=l_lya, vf=l_vf, state=l_state, jac=l_jac,
        )
