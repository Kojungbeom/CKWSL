
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.KSL_base as KSL_base
from torchsummary import summary

import os
import sys

class KeywordSpeakerLocalizer(nn.Module):
    def __init__(self, en_ch):
        super().__init__()
        self.en_ch = en_ch # [[6, 32], [32, 64] ...]
        last_ch = self.en_ch[-1][1]
        self.relu = nn.ReLU()
        self.cls = 1
        self.encoder = KSL_base.KSL_Encoder(self.en_ch)
        self.ksl_Localizer = KSL_base.KSL_Localizer(last_ch)
        self.ksl_Detector = KSL_base.KSL_Detector(last_ch, self.cls)
        
    def forward(self, x):
        out = self.encoder(x)
        out_localizer1, out_localizer2 = self.ksl_Localizer(out)
        out_detector = self.ksl_Detector(out)
        return out_localizer1, out_localizer2, out_detector
    
if __name__ == "__main__":
    encoder_ch = [[6,32],[32,96],[96,192],[192,192]]
    net = KeywordSpeakerLocalizer(encoder_ch)
    print(net)
    sample = torch.randn(1, 6, 16000)
    out_localizer1, out_localizer2, out_detector = net(sample)
    print(f'Output of KSL_Localizer: {out_localizer1.shape}')
    print(f'Output of KSL_Localizer: {out_localizer2.shape}')
    print(f'Output of KSL_Localizer: {out_detector.shape}')
    summary(net.cuda(), (6, 16000))

