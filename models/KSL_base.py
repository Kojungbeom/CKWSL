
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

class KSL_Extractor(nn.Module):
    """Angle discriminator
    """
    def __init__(self, de_ch):
        super().__init__()
        
        self.de_ch = de_ch
        self.k = [4,4,4,4]
        self.s = [2,2,2,2]
        self.p = [1,1,1,1]
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(de_ch[3][1], de_ch[3][0], kernel_size=self.k[0], stride=self.s[0], padding=self.p[0]),
            nn.BatchNorm1d(de_ch[3][0]),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(de_ch[2][1], de_ch[2][0], kernel_size=self.k[1], stride=self.s[1], padding=self.p[1]),
            nn.BatchNorm1d(de_ch[2][0]),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(de_ch[1][1], de_ch[1][0], kernel_size=self.k[2], stride=self.s[2], padding=self.p[2]),
            nn.BatchNorm1d(de_ch[1][0]),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose1d(de_ch[0][1], de_ch[0][0], kernel_size=self.k[3], stride=self.s[3], padding=self.p[3]),
        )
        
    def forward(self, x):
        out = self.decoder1(x)
        out = self.decoder2(out) 
        out = self.decoder3(out) 
        out = self.decoder4(out)
        return out
    
class KSL_Localizer(nn.Module):
    """Angle discriminator
    """
    def __init__(self, last_en_ch):
        super().__init__()
        self.encoder_last = nn.Sequential(
            nn.Conv1d(last_en_ch, last_en_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(last_en_ch),
            nn.ReLU()
        )
        self.globalPooling = nn.AdaptiveMaxPool1d(1)   
        self.fcl = nn.Sequential(
            nn.Linear(last_en_ch, int(last_en_ch//2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(last_en_ch//2), 12)
        )
        
    def forward(self, x):
        #out = self.encoder_last(x)
        out = self.globalPooling(x)
        out = out.view(out.size(0), -1)
        out = self.fcl(out).view(-1, 2, 6)
        return out[:, 0, :], out[:, 1, :]
    
    
class KSL_Detector(nn.Module):
    """Angle discriminator
    """
    def __init__(self, last_en_ch, cls):
        super().__init__()
        self.encoder_last = nn.Sequential(
            nn.Conv1d(last_en_ch, last_en_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(last_en_ch),
            nn.ReLU()
        )
        self.globalPooling = nn.AdaptiveAvgPool1d(1)   
        self.fcl = nn.Sequential(
            nn.Linear(last_en_ch, int(last_en_ch//2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(last_en_ch//2), cls)
        )
        
    def forward(self, x):
        #out = self.encoder_last(x)
        out = self.globalPooling(x)
        out = out.view(out.size(0), -1)
        out = self.fcl(out)
        return out

class KSL_Encoder(nn.Module):
    def __init__(self, en_ch):
        super().__init__()
        self.en_ch = en_ch # [[6, 32], [32, 64] ...]
        last_ch = self.en_ch[-1][1]
        
        self.encoder1 = nn.Sequential(
            nn.Conv1d(en_ch[0][0], en_ch[0][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[0][1]),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(en_ch[1][0], en_ch[1][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[1][1]),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(en_ch[2][0], en_ch[2][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[2][1]),
            nn.ReLU(),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv1d(en_ch[3][0], en_ch[3][1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(en_ch[3][1]),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = self.encoder4(out)
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
        self.se4 = SELayer1D(en_ch[3][1])
    
    
    def forward(self, x):
        out = self.encoder1(x)
        out = self.relu(self.se1(out))
        out = self.encoder2(out)
        out = self.relu(self.se2(out))
        out = self.encoder3(out)
        out = self.relu(self.se3(out))
        out = self.encoder4(out)
        out = self.relu(self.se4(out))
        return out
        
