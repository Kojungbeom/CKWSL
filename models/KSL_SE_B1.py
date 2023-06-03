
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.KSL_base as KSL_base
from torchsummary import summary

import os
import sys


class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class KSL_Localizer(nn.Module):
    """Angle discriminator
    """
    def __init__(self, last_en_ch):
        super().__init__()
        self.globalPooling = nn.AdaptiveMaxPool1d(1)   
        self.fcl = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(last_en_ch, 12)
        )
        
    def forward(self, x):
        out = self.globalPooling(x)
        out = out.view(out.size(0), -1)
        out = self.fcl(out).view(-1, 2, 6)
        return out[:, 0, :], out[:, 1, :]
    
    
class KSL_Detector(nn.Module):
    """Angle discriminator
    """
    def __init__(self, last_en_ch, cls):
        super().__init__()
        self.globalPooling = nn.AdaptiveMaxPool1d(1)   
        self.fcl = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(last_en_ch, cls)
        )
        
    def forward(self, x):
        out = self.globalPooling(x)
        out = out.view(out.size(0), -1)
        out = self.fcl(out)
        return out

    
    
class KSL_SE_Encoder(nn.Module):
    def __init__(self, en_ch):
        super().__init__()
        self.en_ch = en_ch # [[6, 32], [32, 64] ...]
        last_ch = self.en_ch[-1][1]
        self.relu = nn.ReLU()
        self.encoder1 = nn.Sequential(
            nn.Conv1d(en_ch[0][0], en_ch[0][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[0][1]),
        )
        self.se1 = SELayer1D(en_ch[0][1])
        
        self.encoder2 = nn.Sequential(
            nn.Conv1d(en_ch[1][0], en_ch[1][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[1][1]),
        )
        self.se2 = SELayer1D(en_ch[1][1])
        
        self.encoder3 = nn.Sequential(
            nn.Conv1d(en_ch[2][0], en_ch[2][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[2][1]),
        )
        self.se3 = SELayer1D(en_ch[2][1])
        
        self.encoder4 = nn.Sequential(
            nn.Conv1d(en_ch[3][0], en_ch[3][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[3][1]),
        )
        self.se4_1 = SELayer1D(en_ch[3][1])
        self.se4_2 = SELayer1D(en_ch[3][1])
    
    
    def forward(self, x):
        out = self.encoder1(x)
        out = self.relu(self.se1(out))
        out = self.encoder2(out)
        out = self.relu(self.se2(out))
        out = self.encoder3(out)
        out = self.relu(self.se3(out))
        out = self.encoder4(out)
        out1 = self.relu(self.se4_1(out))
        out2 = self.relu(self.se4_2(out))
        return out1, out2

class KeywordSpeakerLocalizerSE_B1(nn.Module):
    def __init__(self, en_ch):
        super().__init__()
        self.en_ch = en_ch # [[6, 32], [32, 64] ...]
        last_ch = self.en_ch[-1][1]
        self.relu = nn.ReLU()
        self.cls = 1
        self.encoder = KSL_SE_Encoder(self.en_ch)
        self.ksl_Localizer = KSL_Localizer(last_ch)
        self.ksl_Detector = KSL_Detector(last_ch, self.cls)
        
    def forward(self, x):
        out1, out2 = self.encoder(x)
        out_localizer1, out_localizer2 = self.ksl_Localizer(out1)
        out_detector = self.ksl_Detector(out2)
        return out_localizer1, out_localizer2, out_detector
    
if __name__ == "__main__":
    encoder_ch = [[6,64],[64,128],[128,256],[256,256]]
    net = KeywordSpeakerLocalizerSE_B1(encoder_ch)
    print(net)
    sample = torch.randn(1, 6, 16000)
    out_localizer1, out_localizer2, out_detector = net(sample)
    print(f'Output of KSL_Localizer: {out_localizer1.shape}')
    print(f'Output of KSL_Localizer: {out_localizer2.shape}')
    #print(f'Output of KSL_Localizer: {out_extractor.shape}')
    print(f'Output of KSL_Localizer: {out_detector.shape}')
    summary(net.cuda(), (6, 16000))

