"""
Decoder D — maps the final hidden state h_T^S to an answer.

    y_hat = D(h_T^S)

Also computes the two halting signals needed by Dynamic Phase-Space Routing:
    kappa_t = max_k p(y_k | h_t)   — answer confidence
    eta_t   = H(p(y | h_t))        — prediction entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Two-layer MLP that maps hidden state -> answer logits.
    Also exposes confidence and entropy for the halting controller.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2  = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim)
        Returns:
            logits: (batch, vocab_size)
        """
        x = self.norm(h)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def confidence_and_entropy(self, h: torch.Tensor):
        """
        Compute kappa_t and eta_t for the halting controller.

        Returns:
            kappa: (batch,) — max predicted probability
            eta:   (batch,) — entropy of predicted distribution
        """
        logits = self.forward(h)
        probs  = F.softmax(logits, dim=-1)
        kappa  = probs.max(dim=-1).values
        # Entropy: -sum(p * log(p)), clipped for numerical stability
        eta = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        return kappa, eta
