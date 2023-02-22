import os
from typing import Union

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.nn.parameter import Parameter as Parameter

from utils.task_utils import task_data_size


class ConstrainLoss(nn.Module):
    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        """ """
        super(ConstrainLoss, self).__init__()
        self.cfg = cfg
        target_constrain_val = cfg.inverse_problem.input_boundary.target_val
        self.margin = cfg.inverse_problem.input_boundary.margin
        loss_type = cfg.inverse_problem.input_boundary.type

        self.test(target_constrain_val, loss_type)
        self.target_constrain_val = target_constrain_val

        self.create_constrain_eq()

    def forward(self, input, reduction="mean"):
        loss = torch.abs(self.target_constrain_val - (input * self.eq_array).sum(-1))
        loss = torch.clip(loss - self.margin, min=0)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        else:
            raise Exception("'reduction' value is invalid")

    def create_constrain_eq(self):
        input_dim, _ = task_data_size(self.cfg)
        csv_path = os.path.join(
            self.cfg.general.master_dir,
            f"workspace/setting_files/random_eq_{self.cfg.general.task}.csv",
        )
        eq_data = pd.read_csv(csv_path)
        eq_id = self.cfg.inverse_problem.input_boundary.ConstrainEqId
        eq_array = eq_data.loc[eq_id].values.reshape(1, input_dim)
        self.eq_array = Parameter(torch.tensor(eq_array), requires_grad=False)

    def test(self, target_constrain_val, constrain_type):
        """test function for config"""
        assert abs(target_constrain_val) < 1.0
        assert constrain_type == "Constrain"
