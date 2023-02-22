import os
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor
from torch.nn import functional as functional
from torch.nn.parameter import Parameter as Parameter
from torch.optim import lr_scheduler

from inverse_optimization.boundary_loss import BoundaryLossModel
from inverse_optimization.constrain_loss import ConstrainLoss
from inverse_optimization.projector import Projector
from surrogate_models.sg_models import BiggerSurrogateSimulator
from utils.task_utils import task_data_size


class IV_Model(pl.LightningModule):
    """pytorch lightning module for solving inverse problem"""

    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        super().__init__()
        self.cfg = cfg

        self.fw_model = BiggerSurrogateSimulator(cfg)
        self.fw_model.load_state_dict(
            torch.load(
                os.path.join(
                    cfg.output_dirs.fw_model_dir, cfg.fw_model.model_name + ".pth"
                )
            )
        )
        self.fw_model.eval()
        self.proj_net = Projector(cfg)
        if self.cfg.inverse_problem.method == "NA":
            self.create_learnable_tensor()

        elif self.cfg.inverse_problem.method == "NeuralLagrangian":
            self.num_candidate = self.cfg.inverse_problem.proj.num_leanable_input


        self.loss_func = self.construct_loss_func()
        self.batch_size = cfg.inverse_problem.method_parameters.iv_batch_size
        # dummy loss value
        self.zero_tensor = Parameter(torch.zeros(()), requires_grad=False)

    def dummy_loss_func(self, *args, reduction="mean"):
        """dummy loss function that always output zero"""
        return self.zero_tensor

    def NeuralLagrangian_sampling(
        self, batch_idx, return_loss: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        """
        sampling fcandidates unction for NeuralLagrangian method
        """
        self.proj_net.eval()
        x, _, _ = self.proj_net(batch_idx)

        if self.cfg.inverse_problem.input_boundary.type == "NoMaterial":
            # apply a hard constraint
            x = self.FewFeatureMinimization_Transforms(x)

        if return_loss:
            y_hat = self.fw_model(x)
            loss_val, _ = self.loss_func(x, y_hat, self.y_gt, "none")
            x = [x, loss_val]
        return x

    def NA_sampling(self, return_loss: bool = False) -> Union[Tensor, List[Tensor]]:
        """
        sampling fcandidates unction for NA method
        """
        x = self.learnable_tensor

        if self.cfg.inverse_problem.input_boundary.type == "NoMaterial":
            # apply a hard constraint
            x = self.FewFeatureMinimization_Transforms(x)

        if return_loss:
            y_hat = self.fw_model(x)
            loss_val, _ = self.loss_func(x, y_hat, self.y_gt, "none")
            x = [x, loss_val]
        return x

    def create_learnable_tensor(self):
        """create leranable tensor for Neural Adjoint Method"""
        mode = self.cfg.inverse_problem.method_parameters.optimization_mode
        if mode == "single_target":
            tensor_size = 1
        elif mode in ["batch_target", "single_target_for_multi_solution"]:
            tensor_size = self.cfg.inverse_problem.method_parameters.iv_batch_size
        else:
            raise Exception(
                f'please specify correct "mode". current value:{mode}'.format(mode)
            )
        input_dim, output_dim = task_data_size(self.cfg)
        self.learnable_tensor = Parameter(
            torch.rand([tensor_size, input_dim]) * 2.0 - 1.0
        )
        self.cuv = Parameter(torch.eye(tensor_size), requires_grad=False)

    def create_material_array(self):
        """create material array for hard (no material) constraints"""
        input_dim, _ = task_data_size(self.cfg)
        csv_path = os.path.join(
            self.cfg.general.master_dir,
            f"workspace/setting_files/random_NoMate_{self.cfg.general.task}.csv",
        )
        nm_data = pd.read_csv(csv_path)
        nm_id = self.cfg.inverse_problem.input_boundary.ConstrainNMId

        # num_array : one_hot or  few_hot array that describes which feature to minimize
        nm_array = nm_data.loc[nm_id].values.reshape(1, input_dim).astype(np.float32)
        select_array = 1.0 - nm_array
        min_val_array = nm_array * -1.0
        self.select_array = Parameter(torch.tensor(select_array), requires_grad=False)
        self.min_val_array = Parameter(torch.tensor(min_val_array), requires_grad=False)

    def FewFeatureMinimization_Transforms(self, x: Union[Tensor, Parameter]) -> Tensor:
        """apply hard constraint for a feature to be forced to -1."""
        return x * self.select_array + self.min_val_array

    def construct_loss_func(
        self,
    ) -> Callable[[Tensor, Tensor, Tensor, str], Tuple[Tensor, dict]]:
        """
        define loss function
        """
        self.bloss_func = BoundaryLossModel(self.cfg)
        self.mse_loss_func = functional.mse_loss

        # boundary loss and constraint loss setting
        c_coef = self.cfg.inverse_problem.method_parameters.constrain_loss_coef
        bl_coef = self.cfg.inverse_problem.method_parameters.BL_coef

        # loss function for input space constrain
        if self.cfg.inverse_problem.input_boundary.type == "Constrain":
            self.constrain_loss = ConstrainLoss(self.cfg)
        elif self.cfg.inverse_problem.input_boundary.type == "NoMaterial":
            self.create_material_array()
            self.constrain_loss = self.dummy_loss_func
        elif self.cfg.inverse_problem.input_boundary.type is None:
            self.constrain_loss = self.dummy_loss_func
        else:
            raise Exception(
                "please specify loss type. current value:",
                self.cfg.inverse_problem.input_boundary.type,
            )

        def loss_func(
            input: Tensor, y_hat: Tensor, y: Tensor, reduction: str = "mean"
        ) -> Tuple[Tensor, dict]:
            mse_loss = self.mse_loss_func(y_hat, y, reduction="none")
            bloss = self.bloss_func(input, reduction="none") * bl_coef
            c_loss = self.constrain_loss(input, reduction="none") * c_coef

            if reduction == "mean":
                loss_val = mse_loss.mean() + bloss.mean() + c_loss.mean()
            elif reduction == "none":
                # by data
                loss_val = (
                    mse_loss.mean(dim=-1) + bloss.mean(dim=-1) + c_loss.mean(dim=-1)
                )
            else:
                raise Exception(" reduction is invalid. current value:", reduction)

            loss_dict = {
                "loss": loss_val.mean(),
                "mse_loss": mse_loss.mean(),
                "BLoss": bloss.mean(),
                "Closs": c_loss.mean(),
            }
            return loss_val, loss_dict

        return loss_func

    def forward(self, x):
        return self.fw_model(x)

    def training_step(self, batch, batch_idx):
        y_gt = batch[1]
        if self.cfg.inverse_problem.method == "NA":
            x = self.learnable_tensor

            if self.cfg.inverse_problem.input_boundary.type == "NoMaterial":
                # hard constraint
                x = self.FewFeatureMinimization_Transforms(x)

            y_hat = self.fw_model(x)
            loss_each, metrics = self.loss_func(x, y_hat, y_gt, "none")
            self.proj_net.loss_accumulation(loss_each, self.cuv)

        elif self.cfg.inverse_problem.method == "NeuralLagrangian":
            x, cuv, duv = self.proj_net(batch_idx)
            if self.cfg.inverse_problem.input_boundary.type == "NoMaterial":
                # hard constraint
                x = self.FewFeatureMinimization_Transforms(x)

            y_hat = self.fw_model(x)
            loss_each, metrics = self.loss_func(x, y_hat, y_gt, "none")

            self.proj_net.loss_accumulation(loss_each, cuv)
            self.proj_net.branch_history(loss_each, duv, cuv)

        else:
            raise Exception("method error")

        self.y_gt = y_gt
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        sch = self.lr_schedulers()
        sch.step(metrics["loss"])  # type: ignore

        return metrics

    def training_epoch_end(self, training_step_outputs):
        if self.cfg.inverse_problem.method == "NeuralLagrangian":
            reduction_schedule = self.cfg.inverse_problem.proj.reduction_schedule
            reduction_ratio = reduction_schedule[self.current_epoch]
            if type(reduction_ratio) == int:
                self.proj_net.candidate_selection(reduction_ratio)

    def predict_step(self, batch, batch_idx, return_loss=True):
        if self.cfg.inverse_problem.method == "NA":
            return self.NA_sampling(return_loss=return_loss)
        elif self.cfg.inverse_problem.method == "NeuralLagrangian":
            self.proj_net.eval()
            return self.NeuralLagrangian_sampling(batch_idx, return_loss=return_loss)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.inverse_problem.method_parameters.iv_lr,
            eps=self.cfg.inverse_problem.method_parameters.eps,
        )
        decay_method = self.cfg.inverse_problem.method_parameters.iv_decay_method

        if decay_method == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer=opt,
                mode="min",
                factor=self.cfg.inverse_problem.method_parameters.iv_lr_decay_factor,
                patience=self.cfg.inverse_problem.method_parameters.iv_lr_decay_patience,
                verbose=True,
                threshold=1e-3,
                threshold_mode="rel",
            )
        elif decay_method == "CosineAnnealingLR":
            factor = self.cfg.inverse_problem.method_parameters.iv_lr_decay_factor
            lr = self.cfg.inverse_problem.method_parameters.iv_lr
            min_lr = lr * factor
            steps = self.cfg.inverse_problem.method_parameters.optimization_steps
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=opt, T_max=steps, eta_min=min_lr
            )
        else:
            raise Exception(
                "please specify correct decay method. current value:{decay_method}"
            )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "loss",
            },
        }
