from torch import nn


class NA_ReferenceModel(nn.Module):
    '''
    This model refers https://github.com/BensonRen/AEM_DIM_Bench/blob/main/NA/model_maker.py and https://arxiv.org/pdf/2112.10254.pdf
    '''
    def __init__(self, num_mid_layers:int=7):
        '''
        Args:
            num_mid_layers(int): the number of the layers except input and output
        '''
        super(NA_ReferenceModel, self).__init__()

        assert num_mid_layers > 0
        self.num_mid_layers = num_mid_layers

        self.input_layer = nn.Sequential(
            nn.Linear(5, 700, bias=False),
            nn.BatchNorm1d(700),
            nn.ReLU()
        )

        mid_blocks =[]
        for i in range(num_mid_layers):
                mid_blocks.extend(
                    [
                        nn.Linear(700, 700, bias=False),
                        nn.BatchNorm1d(700),
                        nn.ReLU()
                    ]
                )

        self.mid_blocks = nn.Sequential(*mid_blocks)
        self.fin_layer = nn.Linear(700, 256)


    def forward(self, x):
        hidden = self.input_layer(x)
        hidden = self.mid_blocks(hidden)
        out = self.fin_layer(hidden)

        return out


