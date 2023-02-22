import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from surrogate_models.sg_models import BiggerSurrogateSimulator


class FW_Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.loss_func = nn.MSELoss()
        self.cfg = cfg
        self.best_loss = 1e6
        self.model = BiggerSurrogateSimulator(cfg)
        self.data_logs = []

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

        lr = self.opt.param_groups[0]["lr"]
        self.data_logs.append([avg_loss.detach().cpu().numpy(), lr])

    def configure_optimizers(self):
        # weight decay
        if "weight_decay" in self.cfg.fw_train.keys():
            weight_decay = self.cfg.fw_train.weight_decay
        else:
            weight_decay = 0.0
        # select optimizer
        if self.cfg.fw_train.optimizer == "Adam":
            self.opt = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.fw_train.lr,
                eps=self.cfg.fw_train.eps,
                weight_decay=weight_decay,
            )
        elif self.cfg.fw_train.optimizer == "RAdam":
            self.opt = torch.optim.RAdam(
                self.parameters(),
                lr=self.cfg.fw_train.lr,
                eps=self.cfg.fw_train.eps,
                weight_decay=weight_decay,
            )

        # learning rate decay
        decay_method = self.cfg.fw_train.decay_method
        if decay_method == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer=self.opt,
                mode="min",
                factor=self.cfg.fw_train.decay_factor,
                patience=self.cfg.fw_train.decay_patience,
                verbose=True,
                threshold=1e-2,
                threshold_mode="rel",
            )
        elif decay_method == "CosineAnnealingLR":
            factor = self.cfg.fw_train.decay_factor
            lr = self.cfg.fw_train.lr
            min_lr = lr * factor
            steps = self.cfg.fw_train.max_epochs
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=self.opt, T_max=steps, eta_min=min_lr, verbose=False  # type: ignore
            )
        else:
            raise Exception(
                "please specify correct decay method. current value:{decay_method}"
            )

        return {
            "optimizer": self.opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
