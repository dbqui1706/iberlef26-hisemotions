"""
Asymmetric Loss (ASL) for multi-label classification.
Ridnik et al., 2021 — https://arxiv.org/abs/2009.14119

Drop-in replacement for BCEWithLogitsLoss in WeightedMultiLabelTrainer.
Addresses Recall >> Precision imbalance by:
  - Down-weighting easy negatives  (gamma_neg > 0)
  - Not discounting hard positives (gamma_pos = 0)
  - Clipping negative probabilities to suppress noise (clip)
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Recommended defaults for extreme multi-label imbalance:
        gamma_neg = 4   — heavily penalize easy negatives (FP)
        gamma_pos = 0   — do not discount hard positives (FN)
        clip      = 0.05 — floor negative probability at 0.05, removing very easy negatives
        eps       = 1e-8 — numerical stability for log
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, num_labels) — raw logits BEFORE sigmoid
            targets: (batch, num_labels) — float 0/1 labels
        Returns:
            scalar loss
        """
        probs = torch.sigmoid(logits)

        # --- Asymmetric Clip ---
        # For negative samples: shift probability up by clip to remove easy negatives
        # probs_neg is in [clip, 1], so log(1 - probs_neg) ≤ log(1 - clip)
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)
        else:
            probs_neg = probs

        # --- Cross Entropy for positive and negative ---
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        xs_pos = probs
        xs_neg = 1.0 - probs_neg

        loss_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg  # (batch, num_labels)

        # --- Asymmetric Focusing ---
        # pt: "correct" probability — used to compute modulating factor
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = probs             # p  when y=1
            pt_neg = 1.0 - probs_neg  # 1-p when y=0

            # Modulating factor per sample
            # For y=1: (1-pt_pos)^gamma_pos
            # For y=0: (1-pt_neg)^gamma_neg = probs_neg^gamma_neg
            gamma_t_pos = self.gamma_pos * targets
            gamma_t_neg = self.gamma_neg * (1 - targets)
            pt = pt_pos * targets + pt_neg * (1 - targets)
            gamma_t = gamma_t_pos + gamma_t_neg

            loss *= torch.pow(1.0 - pt, gamma_t)

        loss = -loss  # positive loss value

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'
