"""
Encoder E — maps input token embeddings into the hidden space h_0^S.

h_0^S = E(x)  — this is the fixed semantic anchor used by the Lyapunov
energy function throughout all recursive steps. It encodes what question
is being asked and never changes during recursion.

Architecture: linear projection from the base model's embedding dimension
into the shared hidden dimension d, followed by layer norm.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Projects the base model's last-layer representation of the input
    into the student's working hidden dimension.

    For Qwen2.5-1.5B: input_dim = 1536 (same as hidden_dim, so projection is identity-like)
    For Phi-3.5-mini:  input_dim = 3072
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — last-layer hidden state of input tokens
               from the base student model's encoder pass
        Returns:
            h0: (batch, hidden_dim) — semantic anchor for Lyapunov energy
        """
        return self.norm(self.projection(x))
