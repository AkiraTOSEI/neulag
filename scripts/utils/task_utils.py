from typing import Callable, Tuple, Union

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
from torch import Tensor


def task_data_size(cfg: DictConfig) -> Tuple[int, int]:
    """
    this function outputs task data size
    Args:
        cfg(omegaconf.dictconfig.DictConfig): config
    Outputs:
        input_dim(int): data input dimension
        output_dim(int): data output dimension
    """
    if cfg.general.task == "Chen":
        input_dim = 5
        output_dim = 256
    elif cfg.general.task == "Peurifoy":
        input_dim = 8
        output_dim = 201
    elif cfg.general.task == "Deng":
        input_dim = 14
        output_dim = 2000
    else:
        task = cfg.general.task
        raise Exception(f"specified task name is invalid, {task}")
    return input_dim, output_dim


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
    if task_name == "Chen":
        return X_normalization_Chen
    elif task_name == "Peurifoy":
        return X_normalization_Peurifoy
    elif task_name == "Deng":
        return X_normalization_Deng
    else:
        raise Exception("Normalization type is not defined.")


def define_renormalization(
    task_name: str,
) -> Callable[[Union[Tensor, np.ndarray]], Union[Tensor, np.ndarray]]:
    """define renormalization type"""
    if task_name == "Chen":
        return X_renormalization_Chen
    elif task_name == "Peurifoy":
        return X_renormalization_Peurifoy
    elif task_name == "Deng":
        return X_renormalization_Deng
    else:
        raise Exception("Normalization type is not defined.")
