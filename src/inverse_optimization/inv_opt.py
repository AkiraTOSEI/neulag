import copy
import os
import time
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import Callback

from dataloaders.dataloader import Dataloader4InverseModel
from inverse_optimization.inverse_pl_module import IV_Model
from test_functions.test_funcs import Test_DataShape
from utils.task_utils import Num_GPU_Usage


class ForwardModelFreezeCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        for param in pl_module.fw_model.parameters():
            param.requires_grad = False
        pl_module.fw_model.eval()


def inverse_optimization(cfg: Union[DictConfig, ListConfig], test: bool = False):
    """
    inverse optimization for both Neural Lagrangian method and NA method.
    Args:
      cfg(DictConfig) config
      test(bool) : whether to use test-run-mode
    """

    # initial setting from config
    opt_steps = cfg.inverse_problem.method_parameters.optimization_steps
    iv_batch_size = cfg.inverse_problem.method_parameters.iv_batch_size

    # candidate reduction settings
    if cfg.inverse_problem.method == "NeuralLagrangian":
        reduction_schedule = cfg.inverse_problem.proj.reduction_schedule
        print("<info> reduction schedule:", reduction_schedule)
        if len(reduction_schedule) == 1 and reduction_schedule[0] == 1:
            cfg.inverse_problem.proj.use_candidate_selection = False
            max_epochs = 1
            cfg.inverse_problem.method_parameters.optimization_steps = opt_steps
        else:
            cfg.inverse_problem.proj.use_candidate_selection = True
            cfg.inverse_problem.method_parameters.optimization_steps = opt_steps // len(
                reduction_schedule
            )
            max_epochs = len(reduction_schedule)
    elif cfg.inverse_problem.method == "NA":
        max_epochs = 1
    else:
        raise Exception("method name error")

    # if test mode, optimization steps become smaller
    if test:
        cfg.inverse_problem.method_parameters.optimization_steps = 5

    # initialization
    iv_model = IV_Model(cfg)
    test_loader, targets = Dataloader4InverseModel(cfg)
    num_gpus = Num_GPU_Usage()

    if test:
        Test_DataShape(cfg, [test_loader])

    # inverse problem optimization training and get results
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=num_gpus,
        gradient_clip_val=cfg.inverse_problem.method_parameters.grad_clip,
        callbacks=[ForwardModelFreezeCallback()],
    )
    srt = time.time()
    trainer.fit(iv_model, train_dataloaders=test_loader)
    training_time = time.time() - srt

    # sampling and save
    time_cols, time_results = ["training_time"], [training_time]
    num_eval_sample_list = cfg.inverse_problem.num_eval_sample_list
    for sample_size in num_eval_sample_list:
        # sampling dataloaer
        sampling_cfg = copy.deepcopy(cfg)
        num_batch = int(np.ceil(sample_size / iv_batch_size))
        sampling_cfg.inverse_problem.method_parameters.optimization_steps = num_batch
        sampling_loader, _ = Dataloader4InverseModel(sampling_cfg)
        # sampling and sort
        srt = time.time()
        sampling_results = trainer.predict(dataloaders=sampling_loader)
        opt_input = torch.concat([result[0] for result in sampling_results], dim=0)
        loss_result = torch.concat([result[1] for result in sampling_results], dim=0)
        loss_result, index = torch.sort(loss_result)
        loss_result = loss_result.detach().cpu().numpy()
        opt_input = opt_input[index].detach().cpu().numpy()
        sampling_time = time.time() - srt

        # save optimized solusions
        np.save(
            os.path.join(
                cfg.output_dirs.result_dir, f"optimized_input_size{sample_size}.npy"
            ),
            opt_input,
        )
        np.save(
            os.path.join(cfg.output_dirs.result_dir, f"FW_loss_size{sample_size}.npy"),
            loss_result,
        )
        time_cols.append(f"size{sample_size}_sampling_time")
        time_results.append(sampling_time)

    # save optimization results and original target data.
    np.savez(
        os.path.join(cfg.output_dirs.result_dir, "targets.npz"),
        x_gt=targets[0][:iv_batch_size],
        y_gt=targets[1][:iv_batch_size],
    )

    return time_cols, time_results
