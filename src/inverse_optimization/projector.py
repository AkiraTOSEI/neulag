from typing import Tuple, Union

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.nn.parameter import Parameter as Parameter

from utils.task_utils import task_data_size


class Swish(nn.Module):  # Swish activation
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock4Proj(nn.Module):
    def __init__(self, input_dim, output_dim, noise=False, noise_scale=1.0):
        super().__init__()
        self.noise = noise
        self.noise_scale = noise_scale
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense1 = nn.Linear(input_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.act1 = Swish()
        self.dense2 = nn.Linear(input_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.act2 = Swish()

        if input_dim == output_dim:
            self.dense_s = nn.Identity()
        else:
            self.dense_s = nn.Linear(input_dim, output_dim)

    def forward(self, x, noise=None):
        if self.noise and noise is None:
            noise1 = torch.randn(1, self.input_dim, device=x.device) * self.noise_scale
            noise2 = torch.randn(1, self.output_dim, device=x.device) * self.noise_scale
        elif self.noise and noise is not None:
            noise1, noise2 = noise
        else:
            noise1 = 0.0
            noise2 = 0.0

        out = self.dense1(x) + noise1
        out = self.norm1(out)
        out = self.act1(out)

        out = self.dense2(out) + noise2
        out = self.norm2(out)

        out += self.dense_s(x)

        out = self.act2(out)

        return out


class Projector(nn.Module):
    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        super(Projector, self).__init__()
        self.cfg = cfg

        if cfg.inverse_problem.method == "NA":
            self.loss_accum = 0.0
            self.ema_alpha = 0.9
            self.candidate_loss_history = []
            return

        self.reflecting_config()

        # create alpha, z
        self.create_tensors()

        # building a network
        proj_noise = cfg.inverse_problem.proj.noise_injection
        noise_scale = cfg.inverse_problem.proj.noise_scale
        print(f"<info> Projector proj_noise:{proj_noise}, noise_scale:{noise_scale}")
        # first layer
        mid_layer_dim = cfg.inverse_problem.proj.mid_layer_dim
        self.input_layer = ResBlock4Proj(
            self.hidden_dim, mid_layer_dim, proj_noise, noise_scale
        )
        # middle layers
        mid_blocks = []
        for i in range(self.num_mid_layers):
            mid_blocks.extend(
                [ResBlock4Proj(mid_layer_dim, mid_layer_dim, proj_noise, noise_scale)]
            )
        self.mid_blocks = nn.Sequential(*mid_blocks)
        # final layer
        fw_input_dim, _ = task_data_size(cfg)
        self.fin_layer = ResBlock4Proj(mid_layer_dim, fw_input_dim)

    def reflecting_config(self):
        """initializing the parameters related to config"""
        # projector noise
        self.num_mid_layers = self.cfg.inverse_problem.proj.num_mid_layers
        if self.num_mid_layers >= 0:
            self.hidden_dim = self.cfg.inverse_problem.proj.input_dim
        else:
            fw_input_dim, _ = task_data_size(self.cfg)
            self.hidden_dim = fw_input_dim
        self.iv_batch_size = self.cfg.inverse_problem.method_parameters.iv_batch_size

        # candidate reduction
        self.use_candidate_selection = True
        self.branch_srt = False
        self.loss_accum = 0.0  # Parameter(torch.zeros((1,)),requires_grad=False) # save loss value for sorting
        self.ema_alpha = self.cfg.inverse_problem.proj.momentum_mu
        self.candidate_loss_history = []
        self.dist_loss_history = []
        self.dist_can_history = []
        assert 0.0 <= self.ema_alpha and self.ema_alpha <= 1.0
        assert type(self.hidden_dim) == int

    def batch_candidate(self, batch_index: int):
        """
        Assign z, alpha to mini-batch
        Args:
            batch_index(int): batch index
        Outputs:
            batch_z(Tensor): projector input z, shape = (self.iv_batch_size, self.hidden_dim)
            batch_cand(Tensor): the candidates of NA method, shape = (self.iv_batch_size, fw_input
            batch_gamma(Tensor): the coefficient for projector output, shape = (self.iv_batch_size, fw_input)
            candidate_use_vec(Tensor): one_hot vector indicating which NA candidates to use, shape=(self.iv_batch_size, self.num_cand)
            branch_use_vec(Tensor): : one_hot vector indicating which branches to use, shape=(self.iv_batch_size, self.num_branch)
        """
        idx_cand = batch_index % self.candidate_batch_length  # type: ignore
        idx_branch = batch_index % self.dist_batch_length  # type: ignore
        candidate_use_vec = functional.one_hot(
            self.candidate_index_batch[idx_cand].to(self.learnable_constant.device),
            num_classes=self.num_cand,
        ).unsqueeze(-1)
        batch_cand = torch.sum(candidate_use_vec * self.na_candidates, dim=1)
        candidate_use_vec = candidate_use_vec.squeeze(-1)

        branch_use_vec = functional.one_hot(
            self.dist_index_batch[idx_branch].to(self.learnable_constant.device),
            num_classes=self.num_branch,
        ).unsqueeze(-1)
        batch_gamma = torch.sum(branch_use_vec * self.gamma, dim=1)
        batch_z = torch.sum(branch_use_vec * self.learnable_constant, dim=1)
        branch_use_vec = branch_use_vec.squeeze(-1)
        return batch_z, batch_cand, batch_gamma, candidate_use_vec, branch_use_vec

    def forward(self, batch_index: int) -> Tuple[Tensor, Tensor, Tensor]:
        if self.branch_srt:
            # 2nd step
            # Neural Lagrangian
            batch_z, batch_cand, batch_gamma, cuv, duv = self.batch_candidate(
                batch_index
            )
            # projector propagation
            hidden = self.input_layer(batch_z)
            hidden = self.mid_blocks(hidden)
            hidden = self.fin_layer(hidden)
            # branching
            hidden = hidden * batch_gamma + batch_cand
        else:
            # 1st step
            # same as NA method
            batch_z, batch_cand, batch_gamma, cuv, duv = self.batch_candidate(
                batch_index
            )
            hidden = batch_cand

        out = torch.tanh(hidden)
        return out, cuv, duv

    def create_tensors(self):
        """create candidates ($hat{x}_{c_i}$), coef $alpha$ ,branch latent $z$"""
        fw_input_dim, _ = task_data_size(self.cfg)
        # Neural Adjoint candidate
        self.num_cand = self.cfg.inverse_problem.proj.num_leanable_input
        # Neural Lagrangian Distribution
        self.num_branch = self.cfg.inverse_problem.proj.num_branch

        # calculacte reduction
        reduction = 1
        for _r in self.cfg.inverse_problem.proj.reduction_schedule:
            reduction *= _r
        assert self.num_cand % reduction == 0
        self.num_final_candidates = self.num_cand // reduction

        assert self.iv_batch_size >= self.num_branch
        assert self.num_branch % self.num_final_candidates == 0

        batch_z_dim = self.cfg.inverse_problem.proj.input_dim

        # create distribution candidate index tensor
        self.candidate_index = torch.arange(self.num_cand)
        self.create_candidate_tensor(self.candidate_index)
        # print(f"self.candidate_index_batch : {self.candidate_index_batch}")
        # print(f"self.candidate_batch_length : {self.candidate_batch_length}")

        self.na_candidates = Parameter(
            (torch.rand([1, self.num_cand, fw_input_dim]) * 2.0 - 1)
        )
        self.gamma = Parameter(torch.zeros([1, self.num_branch, fw_input_dim]))
        self.learnable_constant = Parameter(
            (torch.rand([1, self.num_branch, batch_z_dim]) * 2.0 - 1)
        )

    def loss_accumulation(
        self,
        loss_each: torch.Tensor,
        candidate_use_vec: torch.Tensor,
    ):
        """loss accumulation per distribution candidate.
        we use Exponential Moving Average for loss accumlation.
        Args:
            loss_each: batch loss which is not reduced. shape=(BatchSize,)
            candidate_use_vec: one_hot vectors that indicate which candidate to use
        """
        # print(f"loss_each.shape: {loss_each.shape}")
        # print(f"candidate_use_vec.shape:{candidate_use_vec.shape}")
        num_candidate_in_batch = torch.sum(candidate_use_vec, dim=0)
        loss_each = torch.log10(loss_each).unsqueeze(-1)
        with torch.no_grad():
            loss_per_candidate = torch.sum(loss_each * candidate_use_vec, dim=0)
            loss_per_candidate = loss_per_candidate / num_candidate_in_batch
            diff = loss_per_candidate - self.loss_accum
            diff = torch.nan_to_num(diff, posinf=0.0, nan=0.0)
            self.loss_accum += diff * self.ema_alpha

            self.candidate_loss_history.append(
                loss_per_candidate.detach().cpu().numpy()
            )

    def branch_history(
        self,
        loss_each: torch.Tensor,
        branch_use_vec: torch.Tensor,
        candidate_use_vec: torch.Tensor,
    ):
        """loss accumulation per distribution candidate.
        we use Exponential Moving Average for loss accumlation.
        Args:
            loss_each: batch loss which is not reduced. shape=(BatchSize,)
            branch_use_vec: one_hot vectors that indicate which candidate to use
            candidate_use_vec: one_hot vectors that indicate which candidate to use
        """
        # print(f"loss_each.shape: {loss_each.shape}")
        # print(f"candidate_use_vec.shape:{candidate_use_vec.shape}")
        if self.branch_srt:
            num_branch_in_batch = torch.sum(branch_use_vec, dim=0)
            loss_each = torch.log10(loss_each).unsqueeze(-1)
            with torch.no_grad():
                loss_per_dist = torch.sum(loss_each * branch_use_vec, dim=0)
                loss_per_dist = loss_per_dist / num_branch_in_batch

                self.dist_loss_history.append(loss_per_dist.detach().cpu().numpy())
                self.dist_can_history.append(
                    torch.argmax(candidate_use_vec, dim=-1).detach().cpu().numpy()
                )
        else:
            self.dist_loss_history.append([np.nan] * self.num_branch)
            self.dist_can_history.append([np.nan] * self.num_branch)

    def candidate_selection(self, reduction_ratio):
        """Select promising candidates"""
        if self.branch_srt:
            # if branching already started, no further selection
            print("<INFO> No candidate selection")
            return

        print(f"num candidate index {len(self.candidate_index)} ->", end="")

        # selecting promising candidates
        assert len(self.loss_accum.shape) == 1  # type: ignore
        self.candidate_index = torch.argsort(self.loss_accum)[  # type: ignore
            : len(self.candidate_index) // reduction_ratio
        ]
        print(len(self.candidate_index))

        # update candidate tensor for batch
        self.create_candidate_tensor(self.candidate_index)
        # print("candidate index:", self.candidate_index)
        if self.num_final_candidates == len(self.candidate_index):
            self.branch_srt = True
            print("<INFO> branching candidate starts.")

    def create_candidate_tensor(self, candidate_index: torch.Tensor):
        """create candidate index tensor for batch
        Args:
            candidate_index(torch.Tensor) : shape = (num_candidate,)
        """
        assert (
            np.mod(self.iv_batch_size, self.num_cand) == 0
            or np.mod(self.num_cand, self.iv_batch_size) == 0
        )

        if self.iv_batch_size >= len(candidate_index):
            self.candidate_index_batch = torch.cat(
                [candidate_index] * (self.iv_batch_size // len(candidate_index))
            ).view(-1, self.iv_batch_size)
            self.candidate_batch_length = 1
        else:
            self.candidate_index_batch = self.candidate_index.view(
                -1, self.iv_batch_size
            )
            self.candidate_batch_length = len(candidate_index) // self.iv_batch_size
            # print("check:",  len(self.candidate_index_batch), self.candidate_batch_length)
            assert len(self.candidate_index_batch) == self.candidate_batch_length

        dist_index = torch.arange(self.num_branch)
        self.dist_index_batch = torch.cat(
            [dist_index] * (self.iv_batch_size // len(dist_index))
        ).view(-1, self.iv_batch_size)
        self.dist_batch_length = 1
