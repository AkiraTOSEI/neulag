import os
from typing import Callable, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from boundary_loss import BoundaryLossModel
from dataloaders.dataloader import Dataloader4InverseModel
from input_limit_loss import InputLimitLoss
from models.Neural_Adjoint.ADM_model import NA_ReferenceModel_ADM
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import Callback
from test_functions.test_funcs import Test_DataShape
from torch import Tensor
from torch.optim import lr_scheduler
from utils.task_utils import Num_GPU_Usage, task_data_size


class ProjectorModel(nn.Module):
    """
    boundary loss function
    """

    def __init__(self, cfg: DictConfig):
        super(ProjectorModel, self).__init__()
        fw_input_dim, _ = task_data_size(cfg)
        hidden_dim = cfg.inverse_problem.proj.input_dim
        self.num_mid_layers = cfg.inverse_problem.proj.num_mid_layers
        mid_layer_dim = cfg.inverse_problem.proj.mid_layer_dim
        bn_use = cfg.inverse_problem.proj.bn_use

        assert self.num_mid_layers > 0
        assert type(bn_use) == bool
        assert type(hidden_dim) == int

        if bn_use:
            norm = nn.BatchNorm1d
            bias = False
        else:
            norm = nn.Identity
            bias = True

        self.input_layer = nn.Sequential(
            nn.Linear(hidden_dim, mid_layer_dim, bias=bias),
            norm(mid_layer_dim),
            nn.ReLU(),
        )

        mid_blocks = []
        for i in range(self.num_mid_layers):
            mid_blocks.extend(
                [
                    nn.Linear(mid_layer_dim, mid_layer_dim, bias=bias),
                    norm(mid_layer_dim),
                    nn.ReLU(),
                ]
            )

        self.mid_blocks = nn.Sequential(*mid_blocks)
        self.fin_layer = nn.Linear(mid_layer_dim, fw_input_dim)

    def forward(self, z):
        hidden = self.input_layer(z)
        hidden = self.mid_blocks(hidden)
        out = self.fin_layer(hidden)
        return out


class IV_Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.fw_model = self.define_model(cfg.general.task)
        self.fw_model.load_state_dict(
            torch.load(os.path.join(cfg.output_dirs.fw_model_dir, "model.pth"))
        )
        self.fw_model.eval()
        if self.cfg.inverse_problem.method == "NA":
            self.create_learnable_tensor()
        elif self.cfg.inverse_problem.method == "NeuralLagrangian":
            self.proj_net = ProjectorModel(cfg)
            self.hidden_dim = cfg.inverse_problem.proj.input_dim
            self.tensor_size = self.cfg.train.batch_size

        self.loss_func = self.construct_loss_func()
        self.batch_size = cfg.train.batch_size

    def define_model(self, task_name: str):
        if task_name == "Chen":
            return NA_ReferenceModel_Chen(self.cfg)
        elif task_name == "Peurifoy":
            return NA_ReferenceModel_Peurifoy(self.cfg)
        elif task_name == "Deng":
            return NA_ReferenceModel_ADM(self.cfg)
        else:
            task = self.cfg.general.task
            raise Exception(f"specified task name is invalid, {task}")

    def create_learnable_tensor(self):
        """create leranable tensor for Neural Adjoint Method"""
        mode = self.cfg.inverse_problem.method_parameters.optimization_mode
        if mode == "single_target":
            tensor_size = 1
        elif mode in ["batch_target", "single_target_for_multi_solution"]:
            tensor_size = self.cfg.train.batch_size
        else:
            raise Exception(
                f'please specify correct "mode". current value:{mode}'.format(mode)
            )
        input_dim, output_dim = task_data_size(self.cfg)
        self.learnable_tensor = nn.Parameter(torch.rand([tensor_size, input_dim]))

    def forward(self, x):
        return self.fw_model(x)

    def training_step(self, batch, batch_idx):

        if self.cfg.inverse_problem.method == "NA":
            x = self.learnable_tensor
            y_hat = self.fw_model(x)
        elif self.cfg.inverse_problem.method == "NeuralLagrangian":
            z = torch.randn(self.batch_size, self.hidden_dim).to(batch[1].device)
            x = self.proj_net(z)
            y_hat = self.fw_model(x)

        loss, loss_list = self.loss_func(x, y_hat, batch[1])
        metrics = {"loss": loss, "mMSE_loss": loss_list[0], "ill_loss": loss_list[-1]}

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return metrics

    def construct_loss_func(
        self,
    ) -> Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        self.bloss_func = BoundaryLossModel(self.cfg)
        mse_loss_func = nn.functional.mse_loss
        coef = self.cfg.inverse_problem.method_parameters.BL_coef
        if self.cfg.inverse_problem.method == "NeuralLagrangian":
            margin = self.cfg.inverse_problem.method_parameters.margin

        if self.cfg.inverse_problem.input_boundary is not None:
            self.ill_loss_func = InputLimitLoss(self.cfg)

        def NA_loss_func(
            input: Tensor, y_hat: Tensor, y: Tensor
        ) -> Tuple[Tensor, List[Tensor]]:
            mse_loss = mse_loss_func(y_hat, y)
            bloss = self.bloss_func(input)
            ill_loss = self.ill_loss_func(input)
            loss_val = mse_loss + coef * (bloss + ill_loss)
            return loss_val, [mse_loss, bloss, ill_loss]

        def NeuralLagrangian_loss_func(
            input: Tensor, y_hat: Tensor, y: Tensor
        ) -> Tuple[Tensor, List[Tensor]]:
            modified_mse_loss = torch.clip(mse_loss_func(y_hat, y) - margin, min=0)
            bloss = self.bloss_func(input)
            ill_loss = self.ill_loss_func(input)
            loss_val = modified_mse_loss + coef * (bloss + ill_loss)
            return loss_val, [modified_mse_loss, bloss, ill_loss]

        if self.cfg.inverse_problem.method == "NA":
            return NA_loss_func
        elif self.cfg.inverse_problem.method == "NeuralLagrangian":
            return NeuralLagrangian_loss_func
        else:
            raise Exception("Loss Function Error")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.cfg.inverse_problem.method_parameters.lr
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode="min",
            factor=self.cfg.inverse_problem.method_parameters.lr_decay_factor,
            patience=self.cfg.inverse_problem.method_parameters.lr_decay_patience,
            verbose=False,
            # threshold=1e-4,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "loss",
            },
        }

    def NeuralLagrangian_sampling(self) -> torch.Tensor:
        """
        input sampling function for NeuralLagrangian method
        """
        z = torch.randn(self.batch_size, self.hidden_dim)  # .to(batch[1].device)
        x = self.proj_net(z)
        return x


class ForwardModelFreezeCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        for param in pl_module.fw_model.parameters():
            param.requires_grad = False
        pl_module.fw_model.eval()


def backward_optimization(cfg, test=False):
    """
    input optimization with Neural Adjoint method
    Args:
      cfg(omegaconf.dictconfig.DictConfig) config
      test(bool) : whether to use test-run-mode
    """

    # initial setting from config
    bw_steps = cfg.inverse_problem.method_parameters.optimization_steps
    # eval_size = cfg.inverse_problem.eval_batch_size
    iv_model = IV_Model(cfg)
    test_loader, targets = Dataloader4InverseModel(cfg)
    num_gpus = Num_GPU_Usage()

    # if test mode, optimization steps become smaller
    if test:
        bw_steps = 1
        Test_DataShape(cfg, [test_loader])
        # eval_size = 5

    # inverse problem optimization training and get results
    trainer = pl.Trainer(
        max_epochs=bw_steps, gpus=num_gpus, callbacks=[ForwardModelFreezeCallback()]
    )
    trainer.fit(iv_model, train_dataloaders=test_loader)
    if cfg.inverse_problem.method == "NA":
        opt_input = iv_model.learnable_tensor.detach().cpu().numpy()
    elif cfg.inverse_problem.method == "NeuralLagrangian":
        opt_input = iv_model.NeuralLagrangian_sampling().detach().cpu().numpy()

    # save optimization results and original target data.
    np.save(
        os.path.join(cfg.output_dirs.result_dir, "optimized_input.npy"),
        opt_input,
    )
    np.savez(
        os.path.join(cfg.output_dirs.result_dir, "targets.npz"),
        x_gt=targets[0],
        y_gt=targets[1],
    )

    return iv_model
