# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
import __init__

from config import OptInit
from architecture import SparseCADGCN, DetectionLoss
from Datasets.svg import SESYDFloorPlan

import os
import sys
import time
import datetime
import argparse
import numpy as np
import random

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset




import logging

if __name__ == "__main__":
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader ...')
    
    test_dataset = SESYDFloorPlan(opt.data_dir, pre_transform=T.NormalizeScale(), partition = 'train')
    test_loader = DataLoader(test_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=8, 
        collate_fn = InMemoryDataset.collate)   

    with torch.no_grad():
        for i_batch, (data, slices) in enumerate(test_loader):
                
            print(slices)
            image_slice = slices['x']
            label_slice = slices['gt_labels']
            edge_slice = slices['edges']
            raise SystemExit

            print(data.filepath, data.width, data.height)
            for i in range(0, len(image_slice) - 1):
                start = image_slice[i]
                end = image_slice[i + 1]
                is_control_mask = ~data.is_control[start:end].squeeze()
                pos_img = data.pos[start:end][is_control_mask].cuda()
                edge = data.edge[start]
                    
                pos_img[:, 0] *= data.width[i]
                pos_img[:, 1] *= data.height[i]



                start = label_slice[i]
                end = label_slice[i + 1]
                gt_coord_img = data.gt_bbox[start:end]
                gt_cls_img = data.gt_labels[start:end].unsqueeze(1)


                    
