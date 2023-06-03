import numpy as np
import torch
import torch.nn as nn

def get_at():
    angle_table_base = np.array([
        120.0,
        60.0,
        0.0,
        300.0,
        240.0,
        180.0]
    )

    at = angle_table_base
    at = np.sort(at) 
    return at


def get_label_index(angle, at):
    label_index = 0
    angle = int(angle)
    if (angle < at[1] and angle >= at[0]):
        label_index = 0
        angle = angle / 60
    elif (angle < at[2] and angle >= at[1]):
        label_index = 1
        angle = (angle - at[1]) / 60
    elif (angle < at[3] and angle >= at[2]):
        label_index = 2
        angle = (angle - at[2]) / 60
    elif (angle < at[4] and angle >= at[3]):
        label_index = 3
        angle = (angle - at[3]) / 60
    elif (angle < at[5] and angle >= at[4]):
        label_index = 4
        angle = (angle - at[4]) / 60
    elif (angle < 360 and angle >= at[5]):
        label_index = 5
        angle = (angle - at[5]) / 60
        
    if at[label_index] == angle:
        angle = 0
    if angle == 360:
        angle = 0
    return label_index, angle


def DOA(region_y, angle_y, region_p, angle_p, threshold, class_y, class_p):
    at = get_at()
    doa_err = 0.0
    doa_acc = 0
    residual = 0
    bo = 0
    count = 0
    sig = nn.Sigmoid()
    for ry, ay, rp, ap, cy, cp in zip(region_y, angle_y, region_p, angle_p, class_y, class_p):
        if cy == 1: 
            doa_y = int(at[int(ry)] + ay[int(ry)].item() * 60)

            rp_idx = int(torch.argmax(rp))
            doa_p = int(at[rp_idx] + sig(ap[rp_idx]) * 60)
            #print(doa_p)
            
            if doa_p > 360:
                doa_p = doa_p - 360
            elif doa_p < 0:
                doa_p = 360 + doa_p

            residual = abs(doa_y - doa_p)
            if residual > 180:
                residual = 360 - residual
            doa_err += residual

            if residual <= threshold:
                bo = 1
            else:
                bo = 0
            doa_acc += bo
            count += 1
        else:
            continue
    return doa_err, doa_acc, count