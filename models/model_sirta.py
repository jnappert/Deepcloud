from collections import OrderedDict
import torch.nn as nn
from torch import cat
from layers import model_resnet
"""Same issue with import _C module"""
# from torchvision.models import resnet18, vgg11_bn

from layers.convolutions import ConvBlock, ResBlock
from layers.preprocessing import ImagePreprocessing



class SirtaModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = 1  #channels

        """self.aux_data_model = nn.Sequential(OrderedDict([
            ('aux_fc_1_linear', nn.Linear(8, 16)), #input is 8 for forecasting cause of irradiances
            ('aux_fc_1_act', nn.ReLU()),
            ('aux_fc_2_linear', nn.Linear(16, 16)),
            ('aux_fc_2_act', nn.ReLU()),
            ('aux_fc_3_linear', nn.Linear(16, 16)),
            ('aux_fc_3_act', nn.ReLU()),
        ]))"""

        # Look back 3 images
        """self.aux_data_model = nn.Sequential(OrderedDict([
            ('aux_fc_1_linear', nn.Linear(9, 18)),  # input is 8 for forecasting cause of irradiances
            ('aux_fc_1_act', nn.ReLU()),
            ('aux_fc_2_linear', nn.Linear(18, 18)),
            ('aux_fc_2_act', nn.ReLU()),
            ('aux_fc_3_linear', nn.Linear(18, 18)),
            ('aux_fc_3_act', nn.ReLU()),
        ]))"""

        # Nowcast
        self.aux_data_model = nn.Sequential(OrderedDict([
            ('aux_fc_1_linear', nn.Linear(6, 12)),  # input is 8 for forecasting cause of irradiances
            ('aux_fc_1_act', nn.ReLU()),
            ('aux_fc_2_linear', nn.Linear(12, 12)),
            ('aux_fc_2_act', nn.ReLU()),
            ('aux_fc_3_linear', nn.Linear(12, 12)),
            ('aux_fc_3_act', nn.ReLU()),
        ]))

        # First Convblock should be 4 for shades = Y, set to 2 for SAT
        self.cnn_model_keras = nn.Sequential(OrderedDict([
             ('image_preprocessing', ImagePreprocessing()),
             ('conv_0', ConvBlock(self.channels, 64, stride=2, kernel_size=7, norm='none')),
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
            ('cat_fc_1_linear', nn.Linear(140, 64)), #this is 144 for forecasting, 128 for nowcasting
            ('cat_fc_1_act', nn.ReLU()),
            ('cat_fc_2_linear', nn.Linear(64, 32)),
            ('cat_fc_2_act', nn.ReLU()),
            ('out', nn.Linear(32, 1)),
        ]))

        """self.cnn_model = nn.Sequential(OrderedDict([
             ('image_preprocessing', ImagePreprocessing()),
             ('conv_0', ConvBlock(4, 8)),
             ('res_0', ResBlock(8, 8)),
             ('conv_1', ConvBlock(8, 16, stride=2)),
             ('res_1', ResBlock(16, 16)),
             ('conv_2', ConvBlock(16, 32, stride=2)),
             ('res_2', ResBlock(32, 64)),
             ('conv_3', ConvBlock(64, 128, stride=2)),
             ('res_3', ResBlock(128, 128)),
             ('conv_4', ConvBlock(128, 256, stride=2)),
             ('res_4', ResBlock(256, 256)),
             ('conv_5', ConvBlock(256, 256, stride=2)),
             ('res_5', ResBlock(256, 256)),
             ('avg_pool', nn.AdaptiveAvgPool2d(1)),
             ('flatten', nn.Flatten()),
             ('fc_1_linear', nn.Linear(256, 50)),
             ('fc_1_norm', nn.BatchNorm1d(50)),
             ('fc_1_act', nn.ReLU()),
             ('out', nn.Linear(50, 16)),
         ]))


        self.cat_model = nn.Sequential(OrderedDict([
            ('cat_fc_1_linear', nn.Linear(32, 64)),
            ('cat_fc_2_linear', nn.Linear(64, 1)),
        ]))

        self.cnn_resnet_model = model_resnet.resnet18(pretrained=False, progress=True, num_classes=16)

        self.cat_resnet_model = nn.Sequential(OrderedDict([
            ('cat_fc_1_linear', nn.Linear(32, 64)),
            ('cat_fc_2_linear', nn.Linear(64, 1)),
        ]))"""

        #self.model = nn.Sequential(OrderedDict([
        #    ('image_preprocessing', ImagePreprocessing()),
        #    ('conv_1_1', ConvBlock(1, 4)),
        #    ('conv_1_2', ConvBlock(4, 16)),
        #    ('conv_1_3', ConvBlock(16, 32)),
        #    ('conv_1_4', ConvBlock(32, 64)),
        #    ('conv_2', ConvBlock(64, 128, stride=2)),
        #    ('conv_3', ConvBlock(128, 256, stride=2)),
        #    ('conv_4', ConvBlock(256, 512, stride=2)),
        #    ('avg_pool', nn.AdaptiveAvgPool2d(1)),
        #    ('flatten', nn.Flatten()),
        #    ('fc_1_linear', nn.Linear(512, 100)),
        #    ('fc_1_act', nn.ReLU()),
        # ('out', nn.Linear(100, 1)),
        #]))

    def forward(self, images, aux_data):
        #x1 = self.cnn_resnet_model(images)
        x1 = self.cnn_model_keras(images)

        x2 = self.aux_data_model(aux_data)
        x = cat((x1, x2), dim=1)

        x = self.cat_model_keras(x)
        return x

# add Dropout in resnet block