"""Hybrid Dice + Cross-Entropy loss, matching the proposal's training
objective: L = lambda1 * L_Dice + lambda2 * L_CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multiclass soft Dice loss. Averages the per-class Dice score over all
    classes including background, then returns 1 - mean_dice. `logits` are
    raw model outputs (B, num_classes, H, W); softmax is applied inside.
    """

    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_class.mean()


class DiceCELoss(nn.Module):
    def __init__(self, num_classes, lambda_dice=0.5, lambda_ce=0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, logits, target):
        return self.lambda_dice * self.dice(logits, target) + self.lambda_ce * self.ce(logits, target)


@torch.no_grad()
def mean_dice_score(logits, target, num_classes, smooth=1e-5):
    """Per-class Dice averaged over classes, as a plain metric (not a loss
    -- no gradient, returns the score itself rather than 1 - score).
    """
    preds = logits.argmax(dim=1)
    preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (preds_onehot * target_onehot).sum(dims)
    union = preds_onehot.sum(dims) + target_onehot.sum(dims)
    dice_per_class = (2 * intersection + smooth) / (union + smooth)
    return dice_per_class.mean().item()
