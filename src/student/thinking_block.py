"""
Thinking Block G — the recursive reasoning module.

Implements the update rule from the paper (Eq. 1):
    h_{t+1}^S = h_t^S + G(h_t^S, x)

Architecture (per the paper):
- Gated MLP with layer normalization
- Cross-attention over h_0 at every recursive step
  so the block can re-consult the original question encoding
  and counteract semantic drift proactively

The Lyapunov energy constraint is NOT inside this module —
it is applied as a loss term (L_lya) during training.
This module is purely the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinkingBlock(nn.Module):
    """
    One recursive thinking step: takes h_t and h_0 (the anchor),
    returns the *delta* to add: h_{t+1} = h_t + G(h_t, h_0).

    Components:
    1. Cross-attention  — h_t attends to h_0 to re-anchor to the question
    2. Gated MLP        — transforms attended representation
    3. Layer norms      — stabilise activations at each sub-step
    4. Residual gate    — learned scalar gate controls update magnitude
    """

    def __init__(self, hidden_dim: int, mlp_expansion: int = 4, n_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Cross-attention: h_t (query) attends to h_0 (key/value) ──────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # ── Gated MLP ─────────────────────────────────────────────────────
        inner = hidden_dim * mlp_expansion
        self.gate_proj  = nn.Linear(hidden_dim, inner, bias=False)  # gate
        self.value_proj = nn.Linear(hidden_dim, inner, bias=False)  # value
        self.out_proj   = nn.Linear(inner, hidden_dim, bias=False)
        self.norm_mlp   = nn.LayerNorm(hidden_dim)

        # ── Residual gate: learned scalar in (0,1) controls step size ─────
        # Initialised near 0 so early training is stable (small updates)
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        h_t: torch.Tensor,   # (batch, hidden_dim) — current hidden state
        h_0: torch.Tensor,   # (batch, hidden_dim) — fixed semantic anchor
    ) -> torch.Tensor:
        """
        Returns delta: (batch, hidden_dim)
        Caller adds it: h_{t+1} = h_t + delta
        """
        # Reshape to (batch, seq=1, hidden_dim) for MultiheadAttention
        q = h_t.unsqueeze(1)   # query: current state
        kv = h_0.unsqueeze(1)  # key/value: original question anchor

        # Cross-attention + residual + norm
        attn_out, _ = self.cross_attn(q, kv, kv)
        h_attn = self.norm_attn(h_t + attn_out.squeeze(1))

        # Gated MLP (SwiGLU-style gating)
        gate  = F.silu(self.gate_proj(h_attn))
        value = self.value_proj(h_attn)
        mlp_out = self.out_proj(gate * value)
        delta = self.norm_mlp(mlp_out)

        # Scale by learned gate (sigmoid keeps it in (0,1))
        scale = torch.sigmoid(self.residual_gate)
        return scale * delta
