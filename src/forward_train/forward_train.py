import os
from typing import Union

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from dataloaders.dataloader import Dataloader4FowardModel
from forward_train.forward_pl_module import FW_Model
from test_functions.test_funcs import Test_DataShape, Test_Model_IO_Size
from utils.config import cfg_add_dirs
from utils.task_utils import Num_GPU_Usage


def train_forward_model(cfg: Union[DictConfig, ListConfig], test: bool = False):
    """train forward model
    Args:
        cfg: config
        test(bool) : whether to use test-run-mode
    """
    # initial setting from config
    cfg_add_dirs(cfg)
    max_epochs = cfg.fw_train.max_epochs
    fw_model_path = os.path.join(
        cfg.output_dirs.fw_model_dir, f"{cfg.fw_model.model_name}.pth"
    )

    if os.path.exists(fw_model_path):
        assert type(cfg.fw_model.model_rewrite) == bool
        if cfg.fw_model.model_rewrite:
            print(f" <info> {cfg.fw_model.model_name} forward model will re-train.")
        else:
            print(
                f" <info> {cfg.fw_model.model_name} forward model is already exists. so skip forward model trainig"
            )
            return

    # forward train initialization
    pl_model = FW_Model(cfg)
    train_loader, valid_loader = Dataloader4FowardModel(cfg)

    # if test mode, 2 test functions are used and training step becomes smaller.
    if test:
        Test_Model_IO_Size(cfg, pl_model)
        Test_DataShape(cfg, [train_loader, valid_loader])
        max_epochs = 1

    # forward train with pytorch lightning
    if "grad_clip" in cfg.fw_train.keys():
        clip_val = cfg.fw_train.grad_clip
        print(f"<info> using gradient cliping. clip val:{clip_val}")
    else:
        clip_val = 0

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=Num_GPU_Usage(),
        gradient_clip_val=clip_val
        # callbacks=[checkpoint_callback],
    )
    trainer.fit(
        pl_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # save training results, model and loss.
    torch.save(pl_model.model.to("cpu").state_dict(), fw_model_path)
    best_loss = pl_model.best_loss.detach().cpu().numpy()  # type: ignore
    df = pd.DataFrame(
        [best_loss],
        columns=[cfg.inverse_problem.method + "-" + cfg.general.task],
        index=["best_validation_loss"],
    )
    df.to_csv(
        os.path.join(
            cfg.output_dirs.fw_model_dir,
            f"{cfg.fw_model.model_name}-ForwardModelResults.csv",
        )
    )
    return pl_model.data_logs
