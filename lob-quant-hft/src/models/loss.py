"""
Loss Functions for LOB Classification
======================================
  - CrossEntropyLoss (baseline)
  - FocalLoss        (Lin et al., 2017) — handles class imbalance
  - LabelSmoothedCE  — regularisation via soft targets
  - ClassBalancedLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha   : class weights tensor of shape (C,) or scalar
    gamma   : focusing parameter; 0 = standard CE
    reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float | list[float] | None = None,
        gamma: float = 2.0,
        num_classes: int = 3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.full((num_classes,), alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, C) raw model output (before softmax)
        targets : (N,)  integer class labels
        """
        log_probs = F.log_softmax(logits, dim=-1)                   # (N, C)
        probs     = torch.exp(log_probs)                             # (N, C)

        # Gather the log-prob and prob for the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # (N,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # (N,)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothedCE(nn.Module):
    """
    Cross-entropy with label smoothing.

    Loss = (1 - eps) * CE(logits, hard_targets) + eps * KL(logits, uniform)
    """

    def __init__(self, num_classes: int = 3, smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)                    # (N, C)

        # Soft targets
        with torch.no_grad():
            soft = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            soft.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = -(soft * log_probs).sum(dim=-1).mean()
        return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-balanced loss (Cui et al., 2019).

    Reweights each class by effective number of samples:
      EN(n_i) = (1 - beta^n_i) / (1 - beta)
    """

    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        loss_type: str = "focal",
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        effective_num = [
            (1.0 - beta ** n) / (1.0 - beta) for n in samples_per_class
        ]
        weights = [1.0 / en for en in effective_num]
        total   = sum(weights)
        weights = [w * len(samples_per_class) / total for w in weights]

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

        if loss_type == "focal":
            self.loss_fn: nn.Module = FocalLoss(
                alpha=weights, gamma=gamma, num_classes=len(samples_per_class)
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(
    loss_type: str = "focal",
    num_classes: int = 3,
    alpha: float = 0.5,
    gamma: float = 2.0,
    smoothing: float = 0.1,
    samples_per_class: list[int] | None = None,
    class_weights: list[float] | None = None,
) -> nn.Module:
    """
    Create a loss function by name.

    Parameters
    ----------
    loss_type : 'ce' | 'focal' | 'label_smoothed_ce' | 'class_balanced'
    """
    if loss_type == "ce":
        w = torch.tensor(class_weights) if class_weights else None
        return nn.CrossEntropyLoss(weight=w)

    if loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)

    if loss_type == "label_smoothed_ce":
        return LabelSmoothedCE(num_classes=num_classes, smoothing=smoothing)

    if loss_type == "class_balanced":
        if samples_per_class is None:
            raise ValueError("samples_per_class required for class_balanced loss")
        return ClassBalancedLoss(samples_per_class=samples_per_class, gamma=gamma)

    raise ValueError(f"Unknown loss type: {loss_type!r}")
