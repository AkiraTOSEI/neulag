from typing import List, Tuple, Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.nn.parameter import Parameter as Parameter

from utils.task_utils import task_data_size


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, style_in_ch, style_out_ch):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.bias_linear = nn.Linear(style_in_ch, style_out_ch)
        self.scale_linear = nn.Linear(style_in_ch, style_out_ch)
        self.eps = 1e-8

    def forward(self, x, style):
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True) + self.eps
        y_bias = (self.bias_linear(style)).unsqueeze(-1)
        y_scale = (self.scale_linear(style)).unsqueeze(-1)
        return y_scale * (x - x_mean) / x_std + y_bias


class ResBlockUp(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, c_in_ch: int, scale_factor=None, size=None
    ):
        super(ResBlockUp, self).__init__()
        self.norm1 = AdaptiveInstanceNorm1d(c_in_ch, in_ch)
        self.relu1 = nn.ReLU()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = AdaptiveInstanceNorm1d(c_in_ch, out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.upsample_s = nn.Upsample(size=size, scale_factor=scale_factor)
        self.conv_s = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, inputs):
        input, c = inputs
        h = self.norm1(input, c)
        h = self.relu1(h)
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.norm2(h, c)
        h = self.conv2(h)

        skip = self.upsample_s(input)
        skip = self.conv_s(skip)

        out = skip + h

        return out, c


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, c_in_ch):
        super(ResBlock, self).__init__()
        self.norm1 = AdaptiveInstanceNorm1d(c_in_ch, in_ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1)
        self.norm2 = AdaptiveInstanceNorm1d(c_in_ch, in_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        # self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        if in_ch == out_ch:
            self.conv_s = nn.Identity()
        else:
            self.conv_s = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        # self.pool_s = nn.AvgPool1d(kernel_size=3, stride=2,padding=1)

    def forward(self, inputs):
        input, c = inputs
        h = self.norm1(input, c)
        h = self.relu1(h)
        h = self.conv1(h)
        h = self.norm2(h, c)
        h = self.relu2(h)
        h = self.conv2(h)

        skip = self.conv_s(input)
        out = h + skip

        return out, c


class ResBlock4Feat(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(ResBlock4Feat, self).__init__()
        self.norm1 = nn.LayerNorm(in_ch)
        self.relu1 = nn.ReLU()
        self.dense1 = nn.Linear(in_ch, out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.relu2 = nn.ReLU()
        self.dense2 = nn.Linear(out_ch, out_ch)

        if in_ch == out_ch:
            self.dense_s = nn.Identity()
        else:
            self.dense_s = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        h = self.norm1(x)
        h = self.relu1(h)
        h = self.dense1(h)
        h = self.norm2(h)
        h = self.relu2(h)
        h = self.dense2(h)

        skip = self.dense_s(x)
        out = skip + h

        return out


def model_parameters(
    cfg: Union[DictConfig, ListConfig]
) -> Tuple[List[int], List[int], List[int]]:
    model_name = cfg.fw_model.model_name
    task = cfg.general.task
    if model_name == "base" and task == "Stack":
        hidden_dims = [256, 128, 64]
        num_ResBlocks = [0, 0, 0]
        FeatBlocks_dims = []
    elif model_name == "small" and task == "Stack":
        hidden_dims = [512, 256, 128, 64]
        num_ResBlocks = [0, 0, 1, 1]
        FeatBlocks_dims = [64, 64]
    elif model_name == "medium" and task == "Stack":
        hidden_dims = [1024, 512, 256, 128]
        num_ResBlocks = [1, 1, 1, 2]
        FeatBlocks_dims = [128, 128, 128]
    elif model_name == "large" and task == "Stack":
        hidden_dims = [2048, 1024, 512, 256, 128]
        num_ResBlocks = [2, 2, 4, 4, 4]
        FeatBlocks_dims = [256, 256, 256]

    elif model_name == "base" and task == "ADM":
        hidden_dims = [256, 128, 64]
        num_ResBlocks = [0, 0, 0]
        FeatBlocks_dims = []
    elif model_name == "small" and task == "ADM":
        hidden_dims = [512, 256, 128, 64]
        num_ResBlocks = [0, 0, 1, 1]
        FeatBlocks_dims = [128, 128]
    elif model_name == "medium" and task == "ADM":
        hidden_dims = [1024, 512, 256, 128]
        num_ResBlocks = [1, 1, 1, 2]
        FeatBlocks_dims = [128, 128, 128]
    elif model_name == "large" and task == "ADM":
        hidden_dims = [2048, 1024, 512, 256, 128]
        num_ResBlocks = [2, 2, 4, 4, 4]
        FeatBlocks_dims = [256, 256, 256]

    elif model_name == "base" and task == "Shell":
        hidden_dims = [256, 128, 64]
        num_ResBlocks = [0, 0, 0]
        FeatBlocks_dims = []
    elif model_name == "small" and task == "Shell":
        hidden_dims = [512, 256, 128, 64]
        num_ResBlocks = [0, 0, 1, 1]
        FeatBlocks_dims = [64, 64]
    elif model_name == "medium" and task == "Shell":
        hidden_dims = [1024, 512, 256, 128, 64]
        num_ResBlocks = [0, 0, 2, 2, 2]
        FeatBlocks_dims = [128, 128, 128]
    elif model_name == "large" and task == "Shell":
        hidden_dims = [1024, 512, 256, 128, 64]
        num_ResBlocks = [0, 0, 2, 2, 2]
        FeatBlocks_dims = []
    else:
        raise Exception("")

    return hidden_dims, num_ResBlocks, FeatBlocks_dims


class BiggerSurrogateSimulator(nn.Module):
    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        """
        Surrogate model
        """
        super(BiggerSurrogateSimulator, self).__init__()
        hidden_dims, num_ResBlocks, FeatBlocks_dims = model_parameters(cfg)
        assert len(hidden_dims) == len(num_ResBlocks)
        feat_size, output_size = task_data_size(cfg)
        first_seq_length = int(output_size / (2 ** (len(hidden_dims) - 1)))
        self.const_tensor = Parameter(torch.ones(1, hidden_dims[0], first_seq_length))

        # Feature transformations
        FeatBlocks_dims = [feat_size] + FeatBlocks_dims
        if len(FeatBlocks_dims) == 1:
            self.FeatLayers = nn.Identity()
        else:
            layers = []
            for fb_i in range(len(FeatBlocks_dims) - 1):
                layers.append(
                    ResBlock4Feat(FeatBlocks_dims[fb_i], FeatBlocks_dims[fb_i + 1])
                )
            self.FeatLayers = nn.Sequential(*layers)
        fout_size = FeatBlocks_dims[-1]

        # main stream ResBlocks
        StageBlocks = []
        for stage_i in range(len(num_ResBlocks) - 1):
            stage_block = []
            # ResBlock for a sgate
            for _ in range(num_ResBlocks[stage_i]):
                stage_block.append(
                    ResBlock(hidden_dims[stage_i], hidden_dims[stage_i], fout_size)
                )
            # ResBlockUp for a sgate
            if stage_i + 2 == len(num_ResBlocks):
                scale_factor, size = None, output_size
            else:
                scale_factor, size = 2, None
            stage_block.append(
                ResBlockUp(
                    hidden_dims[stage_i],
                    hidden_dims[stage_i + 1],
                    fout_size,
                    scale_factor=scale_factor,
                    size=size,
                )
            )
            stage_block = nn.Sequential(*stage_block)
            StageBlocks.append(stage_block)
        self.StageBlocks = nn.Sequential(*StageBlocks)

        fin_ResBlocks = []
        for block_i in range(num_ResBlocks[-1]):
            fin_ResBlocks.append(ResBlock(hidden_dims[-1], hidden_dims[-1], fout_size))
        fin_ResBlocks.append(ResBlock(hidden_dims[-1], 1, fout_size))
        self.FinBlocks = nn.Sequential(*fin_ResBlocks)

    def forward(self, x):
        feat = self.FeatLayers(x)
        h = self.const_tensor
        h, _ = self.StageBlocks((h, feat))
        h, _ = self.FinBlocks((h, feat))
        out = h.squeeze()
        return out
