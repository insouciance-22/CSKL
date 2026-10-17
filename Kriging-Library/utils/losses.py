import torch
import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()

    def forward(self, x: t.Tensor, y: t.Tensor) -> t.float:
        return torch.mean((x - y) ** 2)


class mae_loss(nn.Module):
    def __init__(self):
        super(mae_loss, self).__init__()

    def forward(self, x: t.Tensor, y: t.Tensor) -> t.float:
        return torch.mean(torch.abs(x - y))


class mse_loss_masked(nn.Module):
    def __init__(self):
        super(mse_loss_masked, self).__init__()

    def forward(self, imputation: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        return torch.sum((imputation * mask - target * mask) ** 2) / (torch.sum(mask) + 1e-6)


class mae_loss_masked(nn.Module):
    def __init__(self):
        super(mae_loss_masked, self).__init__()

    def forward(self, imputation: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        return t.sum(t.abs(imputation * mask - target * mask)) / (torch.sum(mask) + 1e-6)


class mse_loss_KITS(nn.Module):
    def __init__(self):
        super(mse_loss_KITS, self).__init__()
        self.criterion = mse_loss_masked()

    def forward(self, inputs, target: t.Tensor, mask: t.Tensor) -> t.float:
        imputation, imputation_cyc, target_cyc, mask = inputs
        loss = self.criterion(imputation, target, mask) + \
            self.criterion(imputation_cyc, target_cyc, torch.ones_like(imputation_cyc).bool())
        return loss


class bce_logits_loss(nn.Module):
    def __init__(self):
        super(bce_logits_loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss

    def forward(self, logits: t.Tensor, lbl: t.Tensor) -> t.float:
        return self.criterion(logits, lbl)