from torch import nn


class NA_ReferenceModel_ADM(nn.Module):
    """
    This model refers https://github.com/BensonRen/AEM_DIM_Bench/blob/main/NA/model_maker.py and https://arxiv.org/pdf/2112.10254.pdf
    """

    def __init__(self, cfg):
        """
        Args:
            num_mid_layers(int): the number of the layers except input and output
        """
        super(NA_ReferenceModel_ADM, self).__init__()

        self.num_mid_layers = 8
        assert self.num_mid_layers > 0
        self.input_layer = nn.Sequential(
            nn.Linear(14, 1500, bias=False), nn.BatchNorm1d(1500), nn.ReLU()
        )

        mid_blocks = []
        for i in range(self.num_mid_layers):
            mid_blocks.extend(
                [nn.Linear(1500, 1500, bias=False), nn.BatchNorm1d(1500), nn.ReLU()]
            )

        self.mid_blocks = nn.Sequential(*mid_blocks)
        self.conv_input_layer = nn.Linear(1500, 1000)

        conv_blocks = [
            nn.ConvTranspose1d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(4, 1, kernel_size=1, stride=1, padding=0),
        ]
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):
        hidden = self.input_layer(x)
        hidden = self.mid_blocks(hidden)
        hidden = self.conv_input_layer(hidden)
        hidden = hidden.unsqueeze(1)
        out = self.conv_blocks(hidden)
        out = out.squeeze(1)

        return out
