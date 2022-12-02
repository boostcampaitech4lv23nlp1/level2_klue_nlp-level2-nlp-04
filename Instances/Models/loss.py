import torch
import torch.nn as nn
from . import focal_loss, label_smoothing


def CrossEntropyLoss(output, target):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(output, target)

def Focal_loss(output, target, gamma):
    loss_func = focal_loss.FocalLoss(gamma=gamma)
    return loss_func(output, target)

def LabelSmoothing(output, target, epsilon):
    loss_func = label_smoothing.CrossEntropywithLabelSmoothing(epsilon=epsilon)
    return loss_func(output, target)