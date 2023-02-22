from typing import Tuple

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
from utils.task_utils import task_data_size


class BoundaryLossModel(nn.Module):
    """
    boundary loss function
    """

    def __init__(self, cfg: DictConfig):
        super(BoundaryLossModel, self).__init__()
        input_dim, output_dim = task_data_size(cfg)

        # input data range is [-1,+1]
        self.data_range = nn.Parameter(
            torch.ones((1, input_dim)) * 2.0, requires_grad=False
        )
        self.data_mean = nn.Parameter(torch.zeros((1, input_dim)), requires_grad=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        loss_val = torch.mean(
            self.relu(torch.abs(input - self.data_mean) - self.data_range * 0.5)
        )
        return loss_val


def BoundaryDict2Array(
    cfg: DictConfig, input_boundary: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """create array for boundary condition loss"""
    input_dim, output_dim = task_data_size(cfg)

    for key, val in input_boundary.items():
        if val is None and key.endswith("min"):
            input_boundary[key] = -1e8
        elif val is None and key.endswith("max"):
            input_boundary[key] = 1e8
    min_array = np.zeros(input_dim, dtype=np.float32)
    max_array = np.zeros(input_dim, dtype=np.float32)
    min_array[:], max_array[:] = np.nan, np.nan
    for key, val in input_boundary.items():
        feat_i = int(key.split("_")[0].replace("x", ""))
        if key.endswith("min"):
            min_array[feat_i] = val
        elif key.endswith("max"):
            max_array[feat_i] = val
    assert not np.isnan(min_array).all()
    assert not np.isnan(max_array).all()
    return min_array[np.newaxis, :], max_array[np.newaxis, :]
