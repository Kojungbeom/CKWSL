
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
import matplotlib.pyplot as plt

import utils.utils_base as uBase
from models.KSL_EFFI import EfficientNet
from dataset_MT_KWS import MC_MT_KWS_Dataset, Overlap_interference

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

encoder_ch = [[6,64],[64,128],[128,256],[256,256]]
net = EfficientNet().cuda()
learning_rate = 0.0005


optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14,18,23], gamma=0.1)

coarse_loss = nn.CrossEntropyLoss(reduction="sum")
fine_loss = nn.MSELoss(reduction="sum")
cls_loss = nn.BCELoss(reduction="sum")
ext_loss = nn.L1Loss(reduction="sum")
b = 0

annotation = "data_all.csv"
overlap = [1.0, 0.5, 0.]
sir = [-10, 0, 10]
target_w = 'right'
OL_SIR = Overlap_interference(overlap=overlap, sir=sir)
composed = transforms.Compose([OL_SIR])
r = 1
train_dat = MC_MT_KWS_Dataset(annotations_file=annotation,
                                    target_word=target_w,
                                    transform=composed,
                                    DType="train",
                                    r=r)
valid_dat = MC_MT_KWS_Dataset(annotations_file=annotation,
                                    target_word=target_w,
                                    transform=composed,
                                    DType="valid",
                                    r=r)

file_labels = pd.read_csv(annotation)
file_labels = file_labels[10000:-10000]
no_target_labels = file_labels[~file_labels['fileName'].str.contains(target_w)]
only_target_labels = file_labels[file_labels['fileName'].str.contains(target_w)]
train_indices = random.sample(range(len(no_target_labels)), len(no_target_labels))

file_labels = pd.read_csv(annotation)
file_labels = file_labels[:10000]
no_target_labels = file_labels[~file_labels['fileName'].str.contains(target_w)]
only_target_labels = file_labels[file_labels['fileName'].str.contains(target_w)]
val_indices = random.sample(range(len(no_target_labels)), len(no_target_labels))

file_labels = pd.read_csv(annotation)
file_labels = file_labels[-10000:]
no_target_labels = file_labels[~file_labels['fileName'].str.contains(target_w)]
only_target_labels = file_labels[file_labels['fileName'].str.contains(target_w)]
test_indices = random.sample(range(len(no_target_labels)), len(no_target_labels))

print(len(train_indices), len(val_indices), len(test_indices))

batch_size = 20
train_dataloader = torch.utils.data.DataLoader(train_dat,
                                               batch_size=batch_size)
valid_dataloader = torch.utils.data.DataLoader(valid_dat,
                                               batch_size=1)


weight_path = "weight/20230126-003Drop/24.pt"

net.load_state_dict(torch.load(weight_path))

def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

class LayerResult:
    def __init__(self, payers, layer_index):
        self.hook = payers[layer_index].register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = relu_fn(output).cpu().data.numpy()
    
    def unregister_forward_hook(self):
        self.hook.remove()



result = LayerResult(net._blocks, 0)

for batch_index, (x_sg, y_sg, y_region, y_angle, y_class, bce, _) in enumerate(valid_dataloader):
    x_sg = x_sg.cuda()
    out_localizer1, out_localizer2, out_detector = net(x_sg)
    break

activations = result.features
print(activations.shape)

fig, axes = plt.subplots(4,4)
for row in range(4):
    for column in range(4):
        plt.imshow(activations[0][row*4+column])
        
#plt.show()
