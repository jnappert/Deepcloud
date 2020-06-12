from collections import OrderedDict
import torch
import torch.nn as nn
from torch import cat
from layers import model_resnet
"""Same issue with import _C module"""
# from torchvision.models import resnet18, vgg11_bn

from layers.convolutions import ConvBlock, ResBlock
from layers.preprocessing import ImagePreprocessing



class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = 4
        self.seq_len = 2
        self.input_dim = 142
        self.hidden_dim = 142

        self.aux_data_model = nn.Sequential(OrderedDict([
            ('aux_fc_1_linear', nn.Linear(7, 14)), #input is 8 for forecasting cause of irradiances
            ('aux_fc_1_act', nn.ReLU()),
            ('aux_fc_2_linear', nn.Linear(14, 14)),
            ('aux_fc_2_act', nn.ReLU()),
            ('aux_fc_3_linear', nn.Linear(14, 14)),
            ('aux_fc_3_act', nn.ReLU()),
        ]))

        # First Convblock should be 4 for shades = Y, set to 2 for SAT
        self.cnn_model_keras = nn.Sequential(OrderedDict([
             ('image_preprocessing', ImagePreprocessing()),
             ('conv_0', ConvBlock(1, 64, stride=2, kernel_size=7, norm='none')),
             ('conv_1', ConvBlock(64, 32, stride=2, kernel_size=7, norm='none')),
             ('res_1', ResBlock(32, 32, kernel_size=5, norm='none')),
             ('conv_2', ConvBlock(32, 32, kernel_size=5, stride=2, norm='none')),
             ('res_2', ResBlock(32, 32, norm='none')),
             ('conv_3', ConvBlock(32, 32, stride=2, norm='none')),
             ('res_3', ResBlock(32, 32, norm='none')),
             ('conv_4', ConvBlock(32, 32, stride=2, norm='none')),
             ('res_5', ResBlock(32, 32, norm='none')),
             ('conv_6', ConvBlock(32, 32, stride=2, norm='none')),
             ('res_6', ResBlock(32, 32, norm='none')),
             ('conv_7', ConvBlock(32, 32, stride=2, norm='none')),
             ('res_7', ResBlock(32, 32, norm='none')),
             ('conv_8', ConvBlock(32, 32, stride=2, norm='none')),
#             ('avg_pool', nn.AdaptiveAvgPool2d(1)),
             ('flatten', nn.Flatten()),
             ('fc_1_linear', nn.Linear(32, 512)),
             ('fc_1_act', nn.ReLU()),
             ('fc_2_linear', nn.Linear(512, 128)),
             ('fc_2_act', nn.ReLU()),
        ]))

        self.cat_model_keras = nn.Sequential(OrderedDict([
            ('cat_fc_1_linear', nn.Linear(144, 64)), #this is 144 for forecasting, 128 for nowcasting
            ('cat_fc_1_act', nn.ReLU()),
            ('cat_fc_2_linear', nn.Linear(64, 32)),
            ('cat_fc_2_act', nn.ReLU()),
            ('out', nn.Linear(32, 1)),
        ]))

        self.lstm = nn.Sequential(OrderedDict([
            ('lstm_layers', nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layers, dropout=0.5))]))

        self.linear = nn.Linear(self.hidden_dim, 1)

        self.linear = nn.Sequential(OrderedDict([
            ('cat_fc_1_linear', nn.Linear(self.hidden_dim, 32)),  # this is 144 for forecasting, 128 for nowcasting
            ('cat_fc_1_act', nn.ReLU()),
            ('cat_fc_2_linear', nn.Linear(32, 1))]))

    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                       torch.zeros(self.layers, self.seq_len, self.hidden_dim))

    def forward(self, image_t_minus, image_t_0, aux_data_t_minus, aux_data_t_0):
        x1 = self.cnn_model_keras(image_t_minus)
        z1 = self.cnn_model_keras(image_t_0)
        x2 = self.aux_data_model(aux_data_t_minus)
        z2 = self.aux_data_model(aux_data_t_0)
        x = cat((x1, x2), dim=1).unsqueeze(0)
        z = cat((z1, z2), dim=1).unsqueeze(0)
        #print(x.size(), z.size())
        input = cat((x, z), 0)
        #print(input.size())
        # LSTM layer
        lstm_out, self.hidden = self.lstm(input)
        pred = self.linear(lstm_out[-1])
        return pred

# add Dropout in resnet block