"""
Self-contained attention module for the RatingNet extension.

Implements additive (Bahdanau-style) attention over BiLSTM outputs.
The module is importable on its own and can be wired into ChessEloPredictor
or applied post-hoc to LSTM hidden states.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive attention over a sequence of hidden states.

    Given a sequence H of shape (batch, seq, hidden_dim), the module computes
    an energy score e_t = v^T tanh(W_h h_t + b) and a weight alpha_t that
    can be used either to build a context vector c = sum_t alpha_t h_t or to
    provide per-step importance for the anomaly detector.

    Args:
        hidden_dim: Dimension of each LSTM output vector.
        attention_dim: Intermediate projection size for the score function.
        query: Optional fixed or learned query. If None, a learned query vector
            is created. The query has shape (attention_dim,).
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64, query: torch.Tensor | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.key_projection = nn.Linear(hidden_dim, attention_dim, bias=True)
        if query is None:
            self.query = nn.Parameter(torch.randn(attention_dim) / math.sqrt(attention_dim))
        else:
            self.query = nn.Parameter(query.clone())

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq, hidden_dim)
            mask: Optional (batch, seq) boolean/tensor mask. True/1 means keep.

        Returns:
            context: (batch, hidden_dim) attention-weighted context vector.
            weights: (batch, seq) per-step attention weights (sum to 1 over seq).
        """
        # Project each time step into attention space: (batch, seq, attention_dim)
        projected = torch.tanh(self.key_projection(hidden_states))
        # Score against the learned query: (batch, seq)
        scores = torch.matmul(projected, self.query)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, seq)
        # Context vector: (batch, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class SelfAttention(nn.Module):
    """Simple scaled dot-product self-attention over LSTM outputs.

    Computes attention weights by treating each time step as both key and
    query, using a learned linear projection. Useful as an alternative to
    Bahdanau attention in ablations.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, heads, seq, seq)
        weights = self.dropout(weights)
        attn_output = torch.matmul(weights, v)  # (batch, heads, seq, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.hidden_dim)
        output = self.out_proj(attn_output)
        # Per-step importance (batch, seq), averaged over heads and query
        # positions, matching BahdanauAttention's weights shape.
        step_importance = weights.mean(dim=(1, 2))
        return output, step_importance
