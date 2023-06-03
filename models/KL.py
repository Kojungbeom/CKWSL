

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.KSL_base as KSL_base
from torchsummary import summary

import os
import sys

class KeywordLocalizer(nn.Module):
    def __init__(self, en_ch, cls):
        super().__init__()
        self.en_ch = en_ch # [[6, 32], [32, 64] ...]
        last_ch = self.en_ch[-1][1]
        self.encoder = KSL_base.KSL_Encoder(self.en_ch)
        self.ksl_Localizer = KSL_base.KSL_Localizer(last_ch)
        
    def forward(self, x):
        out = self.encoder(x)
        out_localizer1, out_localizer2 = self.ksl_Localizer(out)
        return out_localizer1, out_localizer2
    
if __name__ == "__main__":
    encoder_ch =  [[6,32],[32,96],[96,192],[192,192]]
    cls = 1
    net = KeywordLocalizer(encoder_ch, cls)
    print(net)
    sample = torch.randn(1, 6, 16000)
    out_localizer1, out_localizer2 = net(sample)
    print(f'Output of KSL_Localizer: {out_localizer1.shape}')
    print(f'Output of KSL_Localizer: {out_localizer2.shape}')
    summary(net.cuda(), (6, 16000))