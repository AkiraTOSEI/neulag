from typing import Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from utils.task_utils import task_data_size


def Test_Model_IO_Size(cfg: Union[DictConfig, ListConfig], model):
    """model input/output size check
    Args:
        cfg(omegaconf.dictconfig.DictConfig) config
        model(pl.LightningModule) : model
    """
    print("Test_Model_IO_Size", end="")
    input_dim, output_dim = task_data_size(cfg)
    out = model(torch.zeros((cfg.fw_train.batch_size, input_dim)))
    assert out.size() == torch.Size([cfg.fw_train.batch_size, output_dim])
    print("      --OK")


def Test_DataShape(cfg: Union[DictConfig, ListConfig], data_loaders: list):
    """test fucntion for data shape
    Args:
        cfg(omegaconf.dictconfig.DictConfig): config
        dataloaders(list)
    """

    print("Test_DataShape", end="")
    input_dim, output_dim = task_data_size(cfg)

    for loader in data_loaders:
        for x, y in loader:
            break
        if not input_dim == x.size()[-1]:
            raise Exception(
                f"data shape error! input_dim:{input_dim}, x.size()[-1]:{x.size()[-1]}. Both value should be same"
            )
        assert output_dim == y.size()[-1]
        assert x.size()[0] == y.size()[0]

    print("      --OK")
