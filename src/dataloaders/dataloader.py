import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils.task_utils import define_normalization


class Dataset4FowardModel(Dataset):
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
    cfg: Union[DictConfig, ListConfig]
) -> Tuple[DataLoader, DataLoader]:
    """create data loader for foward model training
    Args:
        cfg(DictConfig): config
    Outputs:
        train_loader(DataLoader):training dataloader
        valid_loader(DataLoader):validation dataloader
    """
    if "specify_train_data" in cfg.fw_train.keys():
        file_name = cfg.fw_train.specify_train_data
        print("---------------------------------------------------------------------")
        print(f"<info> train data file : {file_name}_x.csv,  {file_name}_y.csv")
        print("---------------------------------------------------------------------")
    else:
        file_name = "data"

    data_x = np.array(
        pd.read_csv(
            os.path.join(cfg.general.data_dir, f"{file_name}_x.csv"), header=None
        )
    ).astype(np.float32)
    data_y = np.array(
        pd.read_csv(
            os.path.join(cfg.general.data_dir, f"{file_name}_y.csv"),
            header=None,
            delimiter=",",
        )
    ).astype(np.float32)

    num = data_x.shape[0]
    if cfg.general.task in ["Stack", "Shell"]:
        test_size = min(0.2, 10000 / num)
    elif cfg.general.task == "ADM":
        test_size = min(0.2, 2000 / num)
    else:
        raise Exception("task name error")
    print(
        f"<info> train data size:{num*(1.-test_size)}, valid data size:{num*test_size}"
    )

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(
        Dataset4FowardModel(X_train, Y_train, cfg),
        batch_size=cfg.fw_train.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        Dataset4FowardModel(X_valid, Y_valid, cfg),
        batch_size=cfg.fw_train.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader


class Dataset4InverseModel(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cfg: Union[DictConfig, ListConfig],
    ):
        super().__init__()
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        x_norm_func = define_normalization(cfg.general.task)
        X = x_norm_func(X)  # type: ignore
        self.iv_batch_size = cfg.inverse_problem.method_parameters.iv_batch_size
        target_id = cfg.inverse_problem.target_id  # target signal ID in evalu dataset
        mode = cfg.inverse_problem.method_parameters.optimization_mode
        self.opt_steps = cfg.inverse_problem.method_parameters.optimization_steps

        if mode == "batch_target":
            assert target_id is None
            X, Y = X[: self.iv_batch_size], Y[: self.iv_batch_size]
        elif mode == "single_target":
            assert type(target_id) == int
            X = X[target_id : target_id + 1]
            Y = Y[target_id : target_id + 1]

        elif mode == "single_target_for_multi_solution":
            assert type(target_id) == int
            X = X[target_id : target_id + 1]
            X = np.concatenate([X] * self.iv_batch_size, axis=0)
            Y = Y[target_id : target_id + 1]
            Y = np.concatenate([Y] * self.iv_batch_size, axis=0)

        else:
            raise Exception(
                f'please specify correct "mode". current value:{mode}'.format(mode)
            )
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y) * self.opt_steps

    def __getitem__(self, _index):
        index = _index % self.iv_batch_size
        return self.X[index], self.Y[index]


def Dataloader4InverseModel(
    cfg: Union[DictConfig, ListConfig]
) -> Tuple[DataLoader, List[np.ndarray]]:
    """create data loader for foward model training
    Args:
        cfg(DictConfig): config
    Outputs:
        test_loader(DataLoader):test dataloader
        targets:(List[np.ndarray]): ground truth x and y
    """
    mode = cfg.inverse_problem.method_parameters.optimization_mode
    iv_batch_size = cfg.inverse_problem.method_parameters.iv_batch_size
    if mode == "interpolation":
        filename = "interpolation_test"
    else:
        filename = "test"
    test_x = np.array(
        pd.read_csv(
            os.path.join(cfg.general.data_dir, f"{filename}_x.csv"), header=None
        )
    ).astype(np.float32)
    test_y = np.array(
        pd.read_csv(
            os.path.join(cfg.general.data_dir, f"{filename}_y.csv"), header=None
        )
    ).astype(np.float32)
    ds = Dataset4InverseModel(test_x, test_y, cfg)
    test_loader = DataLoader(ds, batch_size=iv_batch_size, shuffle=False)

    targets = [ds.X, ds.Y]

    return test_loader, targets
