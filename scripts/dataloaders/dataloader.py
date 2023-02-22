import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.task_utils import define_normalization


class Dataset4FowardModel(torch.utils.data.Dataset):
    def __init__(self, X, Y, cfg):
        super().__init__()
        self.X, self.Y = X.astype(np.float32), Y.astype(np.float32)
        x_norm_func = define_normalization(cfg.general.task)
        self.X = x_norm_func(self.X)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def Dataloader4FowardModel(
    cfg: DictConfig, random_state: int = 2022
) -> Tuple[DataLoader, DataLoader]:
    """create data loader for foward model training
    Args:
        cfg(DictConfig): config
    Outputs:
        train_loader(DataLoader):training dataloader
        valid_loader(DataLoader):validation dataloader
        test_loader(DataLoader):test dataloader
    """
    data_x = np.array(
        pd.read_csv(os.path.join(cfg.general.data_dir, "data_x.csv"), header=None)
    ).astype(np.float32)
    data_y = np.array(
        pd.read_csv(
            os.path.join(cfg.general.data_dir, "data_y.csv"), header=None, delimiter=","
        )
    ).astype(np.float32)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(
        Dataset4FowardModel(X_train, Y_train, cfg),
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        Dataset4FowardModel(X_valid, Y_valid, cfg),
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader


class Dataset4InverseModel(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, cfg: DictConfig):
        super().__init__()
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        x_norm_func = define_normalization(cfg.general.task)
        X = x_norm_func(X)
        num = cfg.train.batch_size
        target_id = cfg.inverse_problem.target_id  # target signal ID in evalu dataset
        mode = cfg.inverse_problem.method_parameters.optimization_mode

        if mode == "batch_target":
            assert target_id is None
            X, Y = X[:num], Y[:num]
        elif mode == "single_target":
            assert type(target_id) == int
            X = X[target_id : target_id + 1]
            Y = Y[target_id : target_id + 1]
        elif mode == "single_target_for_multi_solution":
            assert type(target_id) == int
            X = X[target_id : target_id + 1]
            X = np.concatenate([X] * num, axis=0)
            Y = Y[target_id : target_id + 1]
            Y = np.concatenate([Y] * num, axis=0)
        else:
            raise Exception(
                f'please specify correct "mode". current value:{mode}'.format(mode)
            )
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def Dataloader4InverseModel(
    cfg: DictConfig, random_state: int = 2022
) -> Tuple[DataLoader, list]:
    """create data loader for foward model training
    Args:
        cfg(DictConfig): config
    Outputs:
        train_loader(DataLoader):training dataloader
        valid_loader(DataLoader):validation dataloader
        test_loader(DataLoader):test dataloader
    """

    test_x = np.array(
        pd.read_csv(os.path.join(cfg.general.data_dir, "test_x.csv"), header=None)
    ).astype(np.float32)
    test_y = np.array(
        pd.read_csv(os.path.join(cfg.general.data_dir, "test_y.csv"), header=None)
    ).astype(np.float32)
    ds = Dataset4InverseModel(test_x, test_y, cfg)
    test_loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
    targets = [ds.X, ds.Y]

    return test_loader, targets
