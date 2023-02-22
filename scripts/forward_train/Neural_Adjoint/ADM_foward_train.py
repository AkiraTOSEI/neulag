import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataloaders.dataloader import Dataloader4FowardModel
from models.Neural_Adjoint.ADM_model import NA_ReferenceModel_ADM
from omegaconf.dictconfig import DictConfig
from test_functions.test_funcs import Test_DataShape, Test_Model_IO_Size
from torch.optim import lr_scheduler
from utils.config import cfg_add_dirs
from utils.task_utils import Num_GPU_Usage


class FW_Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.loss_func = nn.MSELoss()
        self.cfg = cfg
        self.best_loss = 1e6
        self.model = self.define_model(cfg.general.task)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        metrics = {"loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return metrics

    def validation_step(self, batch, batch_idx, prog_bar=True):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        metrics = {"val_loss": loss}
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, logger=False
        )
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x["val_loss"] for x in validation_step_outputs]).mean()
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode="min",
            factor=self.cfg.train.lr_decay_factor,
            patience=self.cfg.train.lr_decay_patience,
            verbose=True,
            threshold=1e-4,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

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


def train_forward_model(cfg: DictConfig, test: bool = False):
    """train forward model
    Args:
        cfg(omegaconf.dictconfig.DictConfig) config
        test(bool) : whether to use test-run-mode
    """
    # initial setting from config
    cfg_add_dirs(cfg)
    max_epochs = cfg.train.max_epochs

    # forward train initialization
    pl_model = FW_Model(cfg)
    train_loader, valid_loader = Dataloader4FowardModel(cfg)

    # if test mode, 2 test functions used and training step becomes smaller.
    if test:
        Test_Model_IO_Size(cfg, pl_model)
        Test_DataShape(cfg, [train_loader, valid_loader])
        max_epochs = 1

    # forward train with pytorch lightning
    num_gpus = Num_GPU_Usage()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        cfg.output_dirs.fw_ckpt_dir, save_top_k=1, monitor="val_loss"
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        max_epochs=max_epochs,
        gpus=num_gpus,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        pl_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # save training results, model and loss.
    torch.save(
        pl_model.model.to("cpu").state_dict(),
        os.path.join(cfg.output_dirs.fw_model_dir, "model.pth"),
    )
    best_loss = pl_model.best_loss.detach().cpu().numpy()
    df = pd.DataFrame(
        [best_loss],
        columns=[cfg.inverse_problem.method + "-" + cfg.general.task],
        index=["best_validation_loss"],
    )
    df.to_csv(os.path.join(cfg.output_dirs.result_dir, "ForwardModelResults.csv"))
