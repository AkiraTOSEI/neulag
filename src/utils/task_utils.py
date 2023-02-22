import os
from typing import Callable, Tuple, Union

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor


def task_data_size(cfg: Union[DictConfig, ListConfig]) -> Tuple[int, int]:
    """
    this function outputs task data size
    Args:
        cfg(omegaconf.dictconfig.DictConfig): config
    Outputs:
        input_dim(int): data input dimension
        output_dim(int): data output dimension
    """
    if cfg.general.task == "Stack":
        input_dim = 5
        output_dim = 256
    elif cfg.general.task == "Shell":
        input_dim = 8
        output_dim = 201
    elif cfg.general.task == "ADM":
        input_dim = 14
        output_dim = 2000
    elif cfg.general.task == "TwoBody":
        input_dim = 4
        output_dim = 2
    else:
        task = cfg.general.task
        raise Exception(f"specified task name is invalid, {task}")
    return input_dim, output_dim


def X_normalization_TwoBody(
    data_x: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    return data_x


def X_normalization_Chen(
    data_x: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    data_x = (data_x - 27.5) / 22.5
    return data_x


def X_renormalization_Chen(
    data_x: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    data_x = data_x * 22.5 + 27.5
    return data_x


def X_normalization_Peurifoy(
    data_x: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    data_x = (data_x - 50.0) / 20.0
    return data_x


def X_renormalization_Peurifoy(
    data_x: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    data_x = data_x * 20.0 + 50.0
    return data_x


def X_normalization_Deng(
    data_x: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    return data_x


def X_renormalization_Deng(
    data_x: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    return data_x


def Num_GPU_Usage():
    if torch.cuda.is_available():
        num_gpus_usage = 1
    else:
        num_gpus_usage = 0
    return num_gpus_usage


def define_normalization(
    task_name: str,
) -> Callable[[Union[Tensor, np.ndarray]], Union[Tensor, np.ndarray]]:
    """define normalization type"""
    if task_name == "Stack":
        return X_normalization_Chen
    elif task_name == "Shell":
        return X_normalization_Peurifoy
    elif task_name == "ADM":
        return X_normalization_Deng
    else:
        raise Exception("Normalization type is not defined.")


def define_renormalization(
    task_name: str,
) -> Callable[[Union[Tensor, np.ndarray]], Union[Tensor, np.ndarray]]:
    """define renormalization type"""
    if task_name == "Stack":
        return X_renormalization_Chen
    elif task_name == "Shell":
        return X_renormalization_Peurifoy
    elif task_name == "ADM":
        return X_renormalization_Deng
    else:
        raise Exception("Normalization type is not defined.")


def experiment_skip(cfg: Union[DictConfig, ListConfig], test_run_mode: bool):
    result_exists = os.path.exists(
        os.path.join(cfg.output_dirs.result_dir, "results.csv")
    )
    if result_exists and cfg.general.skip_done_exp and (test_run_mode is False):
        return True
    else:
        return False
