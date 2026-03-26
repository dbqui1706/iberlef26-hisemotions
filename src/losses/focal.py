"""
Focal Loss with class weights for multi-label classification.
Lin et al., 2017 — https://arxiv.org/abs/1708.02002

Unlike ASL which only penalizes easy negatives (gamma_pos=0),
Focal Loss penalizes BOTH easy positives AND easy negatives equally,
forcing the model to focus on hard examples from all classes.

Combined with pos_weight (class weights) to handle class imbalance.
"""

import torch
import torch.nn as nn


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class positive weights.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        - alpha_t = pos_weight for positive samples, 1.0 for negatives
        - gamma = focusing parameter (0 = standard BCE, 2 = standard focal)
        - p_t = predicted probability for the correct class

    Args:
        gamma: focusing parameter. Higher = more focus on hard examples.
               0.0 = standard BCE, 2.0 = standard focal (recommended)
        pos_weight: per-class weight for positive samples (like BCEWithLogitsLoss)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, num_labels) — raw logits before sigmoid
            targets: (batch, num_labels) — float 0/1 labels
        Returns:
            scalar loss
        """
        # Numerically stable computation using logsigmoid
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        # We compute log(p) = logsigmoid(x), log(1-p) = logsigmoid(-x)
        log_p = torch.nn.functional.logsigmoid(logits)
        log_1_minus_p = torch.nn.functional.logsigmoid(-logits)

        # p_t = probability of correct class
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # BCE loss per element
        bce = -(targets * log_p + (1 - targets) * log_1_minus_p)

        # Apply class weights to positive samples
        if self.pos_weight is not None:
            # weight = pos_weight for positive samples, 1.0 for negatives
            weight = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            bce = bce * weight

        # Apply focal modulation
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
