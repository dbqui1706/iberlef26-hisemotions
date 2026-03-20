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
            logits:  (batch, num_labels) — raw logits before sigmoid
            targets: (batch, num_labels) — float 0/1 labels
        Returns:
            scalar loss
        """
        # --- Sigmoid ---
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
 
        # --- Asymmetric Clip ---
        # Shift xs_neg up by clip to completely remove easy negatives.
        # Only applied to xs_neg (negative branch), does not affect xs_pos.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
 
        # --- Log CE loss ---
        loss = (
            targets       * torch.log(xs_pos.clamp(min=self.eps))
            + (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        )
 
        # --- Asymmetric Focusing ---
        # Focusing weights computed under no_grad (Ridnik et al., 2021 design).
        # Gradients flow only through log terms above, NOT through focusing weights.
        # Allowing gradients through pow() causes 0^0 NaN and other instabilities.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                # pt = "correct" probability: p if y=1, (1-p_neg) if y=0
                pt0 = xs_pos * targets            # p   when y=1, 0 when y=0
                pt1 = xs_neg * (1.0 - targets)    # 1-p when y=0, 0 when y=1
                pt  = pt0 + pt1
 
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
                one_sided_w     = torch.pow(1.0 - pt, one_sided_gamma)
                # no_grad → one_sided_w is constant w.r.t. backward
 
            loss = loss * one_sided_w   # gradient flows through `loss` only, not `one_sided_w`
 
        loss = -loss
 
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'
