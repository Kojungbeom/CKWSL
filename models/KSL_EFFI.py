
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.KSL_base as KSL_base
from torchsummary import summary

import os
import sys
import math


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class Conv1dSamePadding(nn.Conv1d):
    """ 1D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        iw = x.size()[-1]
        kw = self.weight.size()[-1]
        sw = self.stride[-1]
        #print(iw, sw, kw)
        ow = math.ceil(iw / sw)
        pad = max((ow - 1) * self.stride[0] + (kw - 1) * self.dilation[0] + 1 - iw, 0)
        if pad > 0:
            x = F.pad(x, [pad//2, pad - pad //2])
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    #print(type(keep_prob), type(inputs))
    random_tensor += torch.rand([batch_size, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor).cuda()
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    """

    def __init__(self, kernel_size, stride, expand_ratio, input_filters, output_filters, se_ratio, drop_n_add):
        super().__init__()
        
        self._bn_mom = 0.1
        self._bn_eps = 1e-03
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.expand_ratio = expand_ratio
        self.drop_n_add = drop_n_add

        # Filter Expansion phase
        inp = input_filters  # number of input channels
        oup = input_filters * expand_ratio  # number of output channels
        if expand_ratio != 1: # add it except at first block 
            self._expand_conv = Conv1dSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm1d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = kernel_size
        s = stride
        self._depthwise_conv = Conv1dSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise(conv filter by filter)
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm1d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1,int(input_filters * se_ratio))  # input channel * 0.25 ex) block2 => 16 * 0.25 = 4
            self._se_reduce = Conv1dSamePadding(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv1dSamePadding(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = output_filters
        self._project_conv = Conv1dSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm1d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
    
    def forward(self, inputs, drop_connect_rate=0.2):
    
        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool1d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
            
        # Output phase
        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.drop_n_add == True:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Batch norm parameters
        bn_mom = 0.1
        bn_eps = 1e-03

        # stem
        in_channels = 6
        out_channels = 32
        self._conv_stem = Conv1dSamePadding(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm1d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([]) # list 형태로 model 구성할 때
        # stage2 r1_k3_s11_e1_i32_o16_se0.25
        self._blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=1, input_filters=32, output_filters=16, se_ratio=0.25, drop_n_add=False))
        # stage3 r2_k3_s22_e6_i16_o24_se0.25
        self._blocks.append(MBConvBlock(3, 2, 6, 16, 24, 0.25, False))
        self._blocks.append(MBConvBlock(3, 1, 6, 24, 24, 0.25, True))
        # stage4 r2_k5_s22_e6_i24_o40_se0.25
        self._blocks.append(MBConvBlock(5, 2, 6, 24, 40, 0.25, False))
        self._blocks.append(MBConvBlock(5, 1, 6, 40, 40, 0.25, True))
        # stage5 r3_k3_s22_e6_i40_o80_se0.25
        self._blocks.append(MBConvBlock(3, 2, 6, 40, 80, 0.25, False))
        self._blocks.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        self._blocks.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        # stage6 r3_k5_s11_e6_i80_o112_se0.25

        
        self._blocks.append(MBConvBlock(5, 1, 6, 80,  112, 0.25, False))


        # Head 
        in_channels = 112
        out_channels = 200
        self._conv_head = Conv1dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm1d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = 0.2
        self._num_classes = 12
        self._localizer = nn.Linear(out_channels, self._num_classes)
        self._spotter = nn.Linear(out_channels, 1)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):          
            x = block(x)
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        if self._dropout:
            x1 = F.dropout(x, p=self._dropout, training=self.training)
        x_local= self._localizer(x).view(-1, 2, 6)
        x_spot = self._spotter(x)
        
        return x_local[:, 0, :], x_local[:, 1, :], x_spot

    
if __name__ == "__main__":
    encoder_ch = [[6,64],[64,128],[128,256],[256,256]]
    net = EfficientNet().cuda()
    summary(net, (6, 16000))
    sample = torch.randn(1, 6, 16000).cuda()
    out_localizer1, out_localizer2, out_detector = net(sample)
    print(f'Output of EFFI_KSL_Localizer1: {out_localizer1.shape}')
    print(f'Output of EFFI_KSL_Localizer2: {out_localizer2.shape}')
    print(f'Output of EFFI_KSL_Spotter: {out_detector.shape}')
    
