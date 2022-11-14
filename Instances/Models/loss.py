import torch
import torch.nn as nn


def L1_loss(output, target):
    loss_func = nn.L1Loss()
    return loss_func(output, target)


def MSE_loss(output, target):
    loss_func = nn.MSELoss()
    return loss_func(output, target)


def BCEWithLogitsLoss(output, target):
    loss_func = nn.BCEWithLogitsLoss()
    return loss_func(output, target)


def RMSE_loss(output, target):
    loss_func = nn.MSELoss()
    return torch.sqrt(loss_func(output, target))


def HUBER_loss(output, target):
    loss_func = nn.HuberLoss()
    return loss_func(output, target)

def CrossEntropyLoss(output, target):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(output,target)