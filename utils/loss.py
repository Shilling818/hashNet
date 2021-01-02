import torch
import torch.nn as nn


class DHNLoss(nn.Module):
    """
    DHN loss function.
    """

    def __init__(self, lamda):
        super(DHNLoss, self).__init__()
        self.lamda = lamda

    def forward(self, H, S):
        # Inner product
        theta = H @ H.t() / 2

        # log(1+e^z) may be overflow when z is large.
        # We convert log(1+e^z) to log(1 + e^(-z)) + z.
        # metric_loss = (torch.log(1 + torch.exp(-(self.lamda * theta).abs())) + theta.clamp(min=0) -
        # self.lamda * S * theta).mean()
        # modify
        metric_loss = (torch.log(1 + torch.exp(- theta.abs())) + theta.clamp(
            min=0) - S * theta).mean()
        quantization_loss = self.logcosh(H.abs() - 1).mean()

        loss = metric_loss + self.lamda * quantization_loss

        return loss

    def logcosh(self, x):
        return torch.log(torch.cosh(x))
