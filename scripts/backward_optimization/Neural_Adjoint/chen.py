import os
from typing import Callable, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch.optim import lr_scheduler

from backward_optimization.Neural_Adjoint.boudary_loss import BoundaryLossModel
from dataloaders.dataloader_chen import Dataloader4InverseModel
from models.Neural_Adjoint.chen_model import NA_ReferenceModel
from test_functions.test_funcs import Test_DataShape
from utils.task_utils import task_data_size


class IV_Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.fw_model = NA_ReferenceModel()
        self.fw_model.load_state_dict(
            torch.load(os.path.join(cfg.output_dirs.fw_model_dir, "model.pth"))
        )
        self.fw_model.eval()
        self.create_learnable_tensor()
        self.loss_func = self.construct_loss_func()

    def create_learnable_tensor(self):
        """create leranable tensor for Neural Adjoint Method
        """
        mode = self.cfg.inverse_problem.method_parameters.optimization_mode
        if mode == "single_target":
            tensor_size = 1
        else:
            tensor_size = self.cfg.inverse_problem.eval_batch_size
        input_dim, output_dim = task_data_size(self.cfg)

        self.learnable_tensor = nn.Parameter(torch.rand([tensor_size, input_dim]))

    def forward(self, x):
        return self.fw_model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.fw_model(self.learnable_tensor)
        loss, mse_loss = self.loss_func(self.learnable_tensor, y_hat, batch[1])
        metrics = {"loss": loss, "mse_loss": mse_loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def construct_loss_func(
        self,
    ) -> Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        self.bloss_func = BoundaryLossModel(self.cfg)
        mse_loss_func = nn.functional.mse_loss
        coef = self.cfg.inverse_problem.method_parameters.BL_coef

        def loss_func(input: Tensor, y_hat: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
            mse_loss = mse_loss_func(y_hat, y)
            bloss = self.bloss_func(input)
            loss_val = mse_loss + coef * bloss
            return loss_val, mse_loss

        return loss_func

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

    bw_steps = cfg.inverse_problem.method_parameters.optimization_steps
    iv_model = IV_Model(cfg)

    test_loader, targets = Dataloader4InverseModel(cfg)
    if test:
        bw_steps = 1
        Test_DataShape(cfg, [test_loader])

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = 0

    trainer = pl.Trainer(
        max_epochs=bw_steps, gpus=gpus, callbacks=[ForwardModelFreezeCallback()]
    )

    print(iv_model.learnable_tensor.detach().cpu().numpy())
    trainer.fit(iv_model, train_dataloaders=test_loader)
    print(iv_model.learnable_tensor.detach().cpu().numpy())

    opt_input = iv_model.learnable_tensor.detach().cpu().numpy()
    print("opt_input:", opt_input)
    np.save(
        os.path.join(cfg.output_dirs.result_dir, "optimized_input.npy"), opt_input,
    )
    np.savez(
        os.path.join(cfg.output_dirs.result_dir, "targets.npz"),
        x_gt=targets[0],
        y_gt=targets[1],
    )

    return iv_model
