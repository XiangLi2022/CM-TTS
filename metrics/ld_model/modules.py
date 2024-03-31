from functools import partial
from typing import List
from torch import nn
import torch

from .mobilenetv2 import ConvBNActivation, InvertedResidual
from .mobilenetv3 import InvertedResidual as InvertedResidualV3
from .mobilenetv3 import InvertedResidualConfig

STRIDE=3

class Projection(nn.Module):
    def __init__(self, in_dim, hidden_dim, activation, output_type, range_clipping=False):
        super(Projection, self).__init__()
        self.output_type = output_type
        self.range_clipping = range_clipping
        if output_type == "scalar":
            output_dim = 1
            if range_clipping:
                self.proj = nn.Tanh()
        elif output_type == "categorical":
            output_dim = 5
        else:
            raise NotImplementedError("wrong output_type: {}".format(output_type))
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        output = self.net(x)

        # range clipping
        if self.output_type == "scalar" and self.range_clipping:
            return self.proj(output) * 2.0 + 3
        else:
            return output


class MBNetConvBlocks(nn.Module):
    def __init__(self, in_channel, ch_list, dropout_rate, activation):
        super(MBNetConvBlocks, self).__init__()

        num_layers = len(ch_list)
        assert num_layers > 0
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            in_ch = in_channel if layer == 0 else ch_list[layer-1]
            out_ch = ch_list[layer]
            self.convs += [
                torch.nn.Sequential(
                    nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = (3,3), padding = (1,1)),
                    nn.Conv2d(in_channels = out_ch, out_channels = out_ch, kernel_size = (3,3), padding = (1,1)),
                    nn.Conv2d(in_channels = out_ch, out_channels = out_ch, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm2d(out_ch),
                    activation(),
                )
            ]
    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        return x

class MobileNetV2ConvBlocks(nn.Module):
    def __init__(self, first_ch, t_list, c_list, n_list, s_list, output_dim):
        super(MobileNetV2ConvBlocks, self).__init__()
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d
        
        # first layer
        features: List[nn.Module] = [ConvBNActivation(1, first_ch, stride=STRIDE, norm_layer=norm_layer)]
        in_ch = first_ch
        # bottleneck layers
        for t, c, n, s in zip(t_list, c_list, n_list, s_list):
            out_ch = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_ch, out_ch, stride, expand_ratio=t, norm_layer=norm_layer))
                in_ch = out_ch
        # last layer
        features.append(ConvBNActivation(in_ch, output_dim, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        time = x.shape[2]
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (time, 1))
        x = x.squeeze(-1).transpose(1, 2)
        return x

class MobileNetV3ConvBlocks(nn.Module):
    def __init__(self, bneck_confs, output_dim):
        super(MobileNetV3ConvBlocks, self).__init__()

        bneck_conf = partial(InvertedResidualConfig, width_mult=1)
        inverted_residual_setting = [bneck_conf(*b_conf) for b_conf in bneck_confs]

        block = InvertedResidualV3

        # Never tested if a different eps and momentum is needed
        #norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(1, firstconv_output_channels, kernel_size=3, stride=STRIDE, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        
        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = output_dim
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        time = x.shape[2]
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (time, 1))
        x = x.squeeze(-1).transpose(1, 2)
        return x