
from tkinter import X
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.KSL_base as KSL_base
from torchsummary import summary

import os
import sys
import math
import time

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

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
        self._blocks.append(MBConvBlock(3, 2, 6, 16, 24, 0.25, False))
        self._blocks.append(MBConvBlock(3, 1, 6, 24, 24, 0.25, True))
        self._blocks.append(MBConvBlock(3, 2, 6, 24, 40, 0.25, False))
        self._blocks.append(MBConvBlock(3, 1, 6, 40, 40, 0.25, True))
        self._blocks.append(MBConvBlock(3, 2, 6, 40, 80, 0.25, False))
        self._blocks.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        self._blocks.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        # stage3 r2_k3_s22_e6_i16_o24_se0.25
        
        
        self._blocks_ssl = nn.ModuleList([])
        self._blocks_kws = nn.ModuleList([])
        
        #self._blocks_ssl.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=1, input_filters=32, output_filters=16, se_ratio=0.25, drop_n_add=False))
        #self._blocks_ssl.append(MBConvBlock(3, 2, 6, 16, 24, 0.25, False))
        #self._blocks_ssl.append(MBConvBlock(3, 1, 6, 24, 24, 0.25, True))
        #self._blocks_ssl.append(MBConvBlock(3, 2, 6, 24, 40, 0.25, False))
        #self._blocks_ssl.append(MBConvBlock(3, 1, 6, 40, 40, 0.25, True))
        #self._blocks_ssl.append(MBConvBlock(3, 2, 6, 40, 80, 0.25, False))
        #self._blocks_ssl.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        #self._blocks_ssl.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        
        #self._blocks_kws.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=1, input_filters=32, output_filters=16, se_ratio=0.25, drop_n_add=False))
        #self._blocks_kws.append(MBConvBlock(3, 2, 6, 16, 24, 0.25, False))
        #self._blocks_kws.append(MBConvBlock(3, 1, 6, 24, 24, 0.25, True))
        #self._blocks_kws.append(MBConvBlock(3, 2, 6, 24, 40, 0.25, False))
        #self._blocks_kws.append(MBConvBlock(3, 1, 6, 40, 40, 0.25, True))
        #self._blocks_kws.append(MBConvBlock(3, 2, 6, 40, 80, 0.25, False))
        #self._blocks_kws.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        #self._blocks_kws.append(MBConvBlock(3, 1, 6, 80, 80, 0.25, True))
        
        # Head
        in_channels = 80
        out_channels = 112
        """
        self._conv_head = Conv1dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm1d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        """
        self._conv_head_ssl = Conv1dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1_ssl = nn.BatchNorm1d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        
        self._conv_head_kws = Conv1dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1_kws = nn.BatchNorm1d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        
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
    

    def ssl_features(self, x):
        for idx, block in enumerate(self._blocks_ssl):          
            x = block(x)
        return x
    
    def kws_features(self, x):
        # Blocks
        for idx, block in enumerate(self._blocks_kws):          
            x = block(x)
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)
        """
        x = relu_fn(self._bn1(self._conv_head(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
            
        x_local= self._localizer(x).view(-1, 2, 6)
        x_kws = self._spotter(x)
        """
        x_ssl = self.ssl_features(x)
        x_kws = self.kws_features(x)

        # Head
        x_ssl = relu_fn(self._bn1_ssl(self._conv_head_ssl(x_ssl)))
        x_ssl = F.adaptive_max_pool1d(x_ssl, 1).squeeze(-1)
        
        x_kws = relu_fn(self._bn1_kws(self._conv_head_kws(x_kws)))
        x_kws = F.adaptive_max_pool1d(x_kws, 1).squeeze(-1)
        
        
        if self._dropout:
            x_ssl = F.dropout(x_ssl, p=self._dropout, training=self.training)
            x_kws = F.dropout(x_kws, p=self._dropout, training=self.training)
        
        x_local= self._localizer(x_ssl).view(-1, 2, 6)
        x_kws = self._spotter(x_kws)
        
        return x_local[:, 0, :], x_local[:, 1, :], x_kws



class convBlock(nn.Module):
    def __init__(self, kernel_size, stride, input_filters, output_filters, padding):
        super().__init__()
        self.conv = nn.Conv1d(input_filters, output_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(output_filters)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        
class standardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cb1 = convBlock(3, 2, 6, 32, 1)
        self.cb2 = convBlock(3, 1, 32, 32, 1)
        self.cb3 = convBlock(3, 2, 32, 64, 1)
        self.cb4 = convBlock(3, 1, 64, 64, 1)
        self.cb5 = convBlock(3, 2, 64, 96, 1)
        self.cb6 = convBlock(3, 1, 96, 96, 1)
        self.cb7 = convBlock(3, 2, 96, 180, 1)
        self.cb8 = convBlock(3, 1, 180, 180, 1)
        self.cb9 = convBlock(3, 1, 180, 180, 1)
        self.cb10 = convBlock(1, 1, 180, 360, 0)
        
        self.localizer = nn.Linear(360, 12)
        self.spotter = nn.Linear(360, 1)
        
    def forward(self, x):
        out = self.cb1(x)
        out = self.cb2(out)
        out = self.cb3(out)
        out = self.cb4(out)
        out = self.cb5(out)
        out = self.cb6(out)
        out = self.cb7(out)
        out = self.cb8(out)
        out = self.cb9(out)
        out = self.cb10(out)
        out = F.adaptive_max_pool1d(out, 1).squeeze(-1)
        out = F.dropout(out)
        out_local = self.localizer(out).view(-1, 2, 6)
        out_kws = self.spotter(out)
        return out_local[:, 0, :], out_local[:, 1, :], out_kws

if __name__ == "__main__":
    encoder_ch = [[6,64],[64,128],[128,256],[256,256]]
    #net = standardNet()
    net = EfficientNet().cuda()
    net.eval()
    #net = standardNet().cuda()
    summary(net, (6, 16000))
    """
    sample = torch.randn(100, 1, 6, 16000)
    w_sample = torch.randn(100, 1, 6, 16000)
    # warmup
    for s in w_sample:
        _ = net(s)
        
    all = 0
    for i in range(5):
        start = time.time()
        for s in sample:
            _ = net(s)
        all += ((time.time()-start) / 100)
    print(all / 5)
    #out_localizer1, out_localizer2, out_detector = net(sample)
    #_ = net(sample)
    #print(time.time()-start)
    #print(f'Output of EFFI_KSL_Localizer1: {out_localizer1.shape}')
    #print(f'Output of EFFI_KSL_Localizer2: {out_localizer2.shape}')
    #print(f'Output of EFFI_KSL_Spotter: {out_detector.shape}')
    """

