import torch
from boundary_loss import BoundaryDict2Array
from dataloaders.dataloader import Dataloader4InverseModel
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
from utils.task_utils import task_data_size


class InputLimitLoss(nn.Module):
    """
    Input limitation loss
    """

    def __init__(self, cfg: DictConfig):
        super(InputLimitLoss, self).__init__()
        self.cfg = cfg
        input_dim, output_dim = task_data_size(cfg)
        self.Choose_ILL_Type()

    def forward(self, x, reduce=True):
        if self.flg == 0:
            return self.zero_tensor
        elif self.flg == 1:
            return self.DictTypeLossForward(x, reduce)
        elif self.flg == 2:
            return self.EuclideanTypeLossForward(x, reduce)

    def Choose_ILL_Type(self):
        """choose InputLimitLoss type and return a flag"""
        if "input_boundary" not in self.cfg.inverse_problem.keys():
            print("<info>  Not use input-boundary-loss")
            self.flg = 0
            zero_tensor = torch.tensor(
                0,
            )
            self.zero_tensor = nn.Parameter(
                zero_tensor,
                requires_grad=False,
            )
            return

        if type(self.cfg.inverse_problem.input_boundary) in [dict, DictConfig]:
            self.DictTypeLossInit()
            self.flg = 1
        elif type(self.cfg.inverse_problem.input_boundary) == float:
            self.EuclideanTypeLossInit()
            self.flg = 2
        else:
            raise Exception("input-boundary-loss setting is wrong")

    def input_boundary_dict_check(self):
        """check dict-type input boundary"""
        input_dim, output_dim = task_data_size(self.cfg)
        input_boundary = self.cfg.inverse_problem.input_boundary
        if not input_dim * 2 == len(input_boundary.keys()):
            raise Exception(
                "Please specify all input boundaries in cfg.inverse_problem.input_boundary"
            )

    def DictTypeLossInit(self):
        self.input_boundary_dict_check()
        input_boundary = self.cfg.inverse_problem.input_boundary
        min_array, max_array = BoundaryDict2Array(self.cfg, input_boundary)
        self.min_array = nn.Parameter(torch.tensor(min_array), requires_grad=False)
        self.max_array = nn.Parameter(torch.tensor(max_array), requires_grad=False)
        print("<info>  use dict-type input-boundary-loss")

    def DictTypeLossForward(self, x: Tensor, reduce: bool) -> Tensor:
        min_loss = torch.clip(self.min_array - x, min=0)
        max_loss = torch.clip(x - self.max_array, min=0)
        loss = min_loss + max_loss
        return loss

    def EuclideanTypeLossInit(self):
        _, targets = Dataloader4InverseModel(self.cfg)
        self.x_gt = nn.Parameter(torch.tensor(targets[0][0:1]), requires_grad=False)
        min_distance = self.cfg.inverse_problem.input_boundary
        self.min_distance = nn.Parameter(
            torch.tensor(min_distance), requires_grad=False
        )
        print("<info>  use euclidean-type input-boundary-loss")

    def EuclideanTypeLossForward(self, x: Tensor, reduce: bool) -> Tensor:
        dist = x - self.x_gt
        dist = torch.sqrt(torch.square(dist).sum(dim=-1))
        loss = torch.clip(self.min_distance - dist, min=0)
        if reduce:
            return loss.sum(dim=-1).mean()
        else:
            return torch.unsqueeze(loss, dim=-1)
