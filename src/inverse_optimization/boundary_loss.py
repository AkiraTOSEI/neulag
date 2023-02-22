from typing import Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.nn.parameter import Parameter as Parameter

from utils.task_utils import task_data_size


class BoundaryLossModel(nn.Module):
    """
    boundary loss function
    """

    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        super(BoundaryLossModel, self).__init__()
        input_dim, output_dim = task_data_size(cfg)

        # input data range is [-1,+1]
        self.data_range = Parameter(
            torch.ones((1, input_dim)) * 2.0, requires_grad=False
        )
        self.data_mean = Parameter(torch.zeros((1, input_dim)), requires_grad=False)
        self.relu = nn.ReLU()

    def forward(self, input, reduction="mean"):
        loss_val = self.relu(torch.abs(input - self.data_mean) - self.data_range * 0.5)
        if reduction == "none":
            return loss_val
        elif reduction == "mean":
            return loss_val.mean()
        else:
            Exception("reduction type error")
