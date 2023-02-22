import torch
from omegaconf.dictconfig import DictConfig
from utils.task_utils import task_data_size


def Test_Model_IO_Size(cfg: DictConfig, model):
    """model input/output size check
    Args:
        cfg(omegaconf.dictconfig.DictConfig) config
        model(pl.LightningModule) : model
    """
    print("Test_Model_IO_Size", end="")
    input_dim, output_dim = task_data_size(cfg)
    out = model(torch.zeros((cfg.train.batch_size, input_dim)))
    assert out.size() == torch.Size([cfg.train.batch_size, output_dim])
    print("      --OK")


def Test_DataShape(cfg: DictConfig, data_loaders: list):
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
        assert input_dim == x.size()[-1]
        assert output_dim == y.size()[-1]
        assert x.size()[0] == y.size()[0]

    print("      --OK")
