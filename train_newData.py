
from this import d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F
import torchaudio
from torch.utils.data.sampler import SubsetRandomSampler
import os
import datetime
import random
import numpy as np
import pandas as pd
import wandb

import utils.utils_base as uBase
from models.KSL_EFFI_v3 import EfficientNet
from dataset_MT_newData import new_Dataset

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

encoder_ch = [2,2]
net = EfficientNet().cuda()
#net = standardNet().cuda()
learning_rate = 0.0005


optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[23, 30], gamma=0.1)

coarse_loss = nn.CrossEntropyLoss(reduction="sum")
fine_loss = nn.MSELoss(reduction="sum")
cls_loss = nn.BCELoss(reduction="sum")
ext_loss = nn.L1Loss(reduction="sum")
b = 0

annotation = "newData3.csv"

train_annotation = 'newData3_train.csv'
valid_annotation = 'newData3_valid.csv'
test_annotation = 'newData3_test.csv'

train_dataset = new_Dataset(annotations_file=train_annotation)
valid_dataset = new_Dataset(annotations_file=valid_annotation)
test_dataset = new_Dataset(annotations_file=test_annotation)


batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=10,
                                               shuffle=False)

t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")[:-3] + "newData"
weight_dir = 'weight/' + t
sample_dir = 'samples/' + t

try:
    os.mkdir(weight_dir)
    os.mkdir(sample_dir)
    os.mkdir(sample_dir + '_val')
except:
    print("Continue")

# 152
accuracy = 0.0
loss = 0
CoarW = 1.
FineW = 1.
ExtW = 0.
ClsW = 1.
b1 = 0

coeff_total = CoarW + FineW + ExtW + ClsW
CoarW = CoarW / coeff_total
FineW = FineW / coeff_total
ExtW = ExtW / coeff_total
ClsW = ClsW / coeff_total
epochs = 35

wandb.init(
    project="KP", 
    name=f"{t}", 
    config={
        "learning_rate": learning_rate,
        "architecture": "standardNet",
        "coar fine cls ext": [CoarW, FineW, ClsW, ExtW], 
        "Date": t + "_" + 'newData',
        "epochs": epochs,
        "encoder_ch": encoder_ch
})

for epoch in range(1, epochs+1):
    net.train()
    running_loss = 0.0
    tloss_ext = 0.
    tloss_cls = 0.
    tloss_coar = 0.
    tloss_fine = 0.
    total_loss = 0.0
    
    for batch_index, (x_sg, y_region, y_angle, y_class, bce, _) in enumerate(train_dataloader):
        
        x_sg = x_sg.cuda()

        y_region = y_region.cuda()
        y_angle = y_angle.cuda()
        y_class = y_class.cuda()
        bce = bce.cuda()

        optimizer.zero_grad()

        out_localizer1, out_localizer2, out_detector = net(x_sg)
        
        loss_cls = cls_loss(torch.sigmoid(out_detector), y_class.unsqueeze(1))
        loss_coarse = coarse_loss(y_class.unsqueeze(1) * out_localizer1, (y_class * y_region).long())
        loss_fine = fine_loss(y_class.unsqueeze(1) * bce * torch.sigmoid(out_localizer2), y_class.unsqueeze(1) * y_angle)

        loss = ClsW * loss_cls + \
            CoarW * loss_coarse + \
            FineW * loss_fine 
        loss.backward()
        optimizer.step()
        

        tloss_cls += loss_cls.item()
        tloss_coar += loss_coarse.item()
        tloss_fine += loss_fine.item()
        
        running_loss += loss.item()
        if batch_index % 10 == 9:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                running_loss / (10 * batch_size),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * batch_size + len(x_sg),
                total_samples=len(train_dataset)
            ))
            running_loss = 0.0
            wandb.log({"train/Train_loss_coar": tloss_coar / (batch_size * 10),
               "train/Train_loss_fine": tloss_fine / (batch_size * 10),
               "train/Train_loss_cls": tloss_cls / (batch_size * 10)},
                      step = b1)
            tloss_ext = 0.0
            tloss_cls = 0.0
            tloss_coar = 0.0
            tloss_fine = 0.0
            b1 += 1
        
    
    scheduler.step()
    
    net.eval()
    accuracy = 0.0
    class_acc = 0.0
    doa_err = 0.0
    doa_acc = 0.0
    count = 0.0
    tloss_ext = 0.
    tloss_cls = 0.
    tloss_coar = 0.
    tloss_fine = 0.
    total_loss = 0.0
    for batch_index, (x_sg, y_region, y_angle, y_class, bce, _) in enumerate(valid_dataloader):
        x_sg = x_sg.cuda()
        y_region = y_region.cuda()
        y_angle = y_angle.cuda()
        y_class = y_class.cuda()
        bce = bce.cuda()

        out_localizer1, out_localizer2, out_detector = net(x_sg)
        
        loss_cls = cls_loss(torch.sigmoid(out_detector), y_class.unsqueeze(1))
        loss_coarse = coarse_loss(y_class.unsqueeze(1) * out_localizer1, (y_class * y_region).long())
        loss_fine = fine_loss(y_class.unsqueeze(1) * bce * torch.sigmoid(out_localizer2), y_class.unsqueeze(1) * y_angle)
        
        tloss_cls += loss_cls.item()
        tloss_coar += loss_coarse.item()
        tloss_fine += loss_fine.item()
        
        loss = ClsW * loss_cls + \
            CoarW * loss_coarse + \
            FineW * loss_fine 
        
        total_loss += loss.item()
        _, region_pidx = out_localizer1.max(1)
        accuracy += region_pidx.eq(y_region).sum()
        
        class_acc += (torch.sigmoid(out_detector) > 0.5).squeeze(1).eq(y_class).sum()
        a, b, c = uBase.DOA(y_region, y_angle, out_localizer1, out_localizer2, 20, y_class, torch.sigmoid(out_detector))
        
        doa_err += a
        doa_acc += b
        count += c
        
    wandb.log({"val/Val_loss_coar": tloss_coar / int(len(valid_dataset)),
               "val/Val_loss_fine": tloss_fine / int(len(valid_dataset)),
               "val/Val_loss_cls": tloss_cls / int(len(valid_dataset)),
               "val/Val_Cls_acc": class_acc.float() / (len(valid_dataset)),
               "val/Val_doa_acc": doa_acc / count,
               "val/Val_doa_err": doa_err / count},
             commit = False)
    
    print('Val Result: Acc: {:0.4f}, C_ACC: {:0.4f}, DOA: {:0.4f}, ACC_k: {:0.4f}'.format(
        accuracy.float() / (len(valid_dataset)),
        class_acc.float() / (len(valid_dataset)),
        doa_err / count,
        doa_acc / count
    ))
    print(f'ext:{round(tloss_ext/(len(valid_dataset)), 6)}, cls:{round(tloss_cls/(len(valid_dataset)), 6)}, coar:{round(tloss_coar/(len(valid_dataset)), 6)}, fine:{round(tloss_fine/(len(valid_dataset)), 6)},')
    weight_path = os.path.join(weight_dir, str(epoch) + '.pt')
    torch.save(net.state_dict(), weight_path, _use_new_zipfile_serialization=False)






