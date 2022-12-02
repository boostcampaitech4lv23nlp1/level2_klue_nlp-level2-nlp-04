import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropywithLabelSmoothing(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target):
        assert preds.size(0) == target.size(0)
        K = preds.size(-1)
        log_probs = F.log_softmax(preds, dim=-1)
        avg_log_probs = (-log_probs).sum(-1).mean()
        ce_loss = F.nll_loss(log_probs, target)
        ce_loss_w_soft_label = (1-self.epsilon) * ce_loss + self.epsilon / K * (avg_log_probs)
        return ce_loss_w_soft_label