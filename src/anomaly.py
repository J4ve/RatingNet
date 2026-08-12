"""
Anomaly detection module for the RatingNet extension.

The module computes a per-move deviation signal between the move-by-move
predicted rating curve and an established player baseline, then weights each
deviation by the attention weight produced by the attention module. Both
per-move and aggregated suspicion scores are exposed.
"""

import torch
import torch.nn as nn


class AnomalyDetector(nn.Module):
    """Per-move and aggregate anomaly scoring.

    Given a per-move rating prediction curve R_t (batch, seq, 2) and a baseline
    rating for each player (batch, 2), the detector computes a deviation
    d_t = |R_t - R_baseline| and an attention-weighted suspicion score
    S = sum_t alpha_t * d_t. The attention weights alpha_t are provided by the
    attention module; when attention is disabled or unavailable a uniform weight
    of 1 / seq_length is used so the aggregate reduces to the mean deviation.

    Args:
        normalize: If True, divide the aggregate score by the sum of attention
            weights (softmax already sums to 1, so this is a safety no-op).
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        predictions: torch.Tensor,
        baseline: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: (batch, seq, 2) standardized or original-scale ratings.
            baseline: (batch, 2) established baseline rating for white and black.
            attention_weights: Optional (batch, seq) attention weights. If None,
                uniform weights are used.

        Returns:
            Dictionary with keys:
                - "per_move_deviation": (batch, seq, 2)
                - "per_move_weighted": (batch, seq, 2)
                - "white_deviation": (batch, seq)
                - "black_deviation": (batch, seq)
                - "white_score": (batch,) aggregated suspicion for white
                - "black_score": (batch,) aggregated suspicion for black
                - "combined_score": (batch,) sum of white and black scores
        """
        batch, seq, _ = predictions.shape
        baseline = baseline.unsqueeze(1)  # (batch, 1, 2)
        per_move_deviation = torch.abs(predictions - baseline)  # (batch, seq, 2)

        if attention_weights is None:
            attention_weights = torch.ones(batch, seq, device=predictions.device, dtype=predictions.dtype) / seq
        else:
            attention_weights = attention_weights.to(predictions.device)
            # Normalize safety check: softmax sums to 1, but if masked we re-normalize.
            if self.normalize:
                weight_sum = attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                attention_weights = attention_weights / weight_sum

        weighted = per_move_deviation * attention_weights.unsqueeze(-1)  # (batch, seq, 2)
        white_score = weighted[:, :, 0].sum(dim=1)  # (batch,)
        black_score = weighted[:, :, 1].sum(dim=1)  # (batch,)

        return {
            "per_move_deviation": per_move_deviation,
            "per_move_weighted": weighted,
            "white_deviation": per_move_deviation[:, :, 0],
            "black_deviation": per_move_deviation[:, :, 1],
            "white_score": white_score,
            "black_score": black_score,
            "combined_score": white_score + black_score,
            "attention_weights": attention_weights,
        }
