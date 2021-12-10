# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
import __init__

from config import OptInit
from architecture import SparseCADGCN, DetectionLoss
from Datasets.svg import SESYDFloorPlan
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.det_util import non_max_suppression, get_batch_statistics, ap_per_class
from torch.nn import functional as F

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


import logging

if __name__ == "__main__":
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader ...')

    if opt.graph == 'bezier':
        from  Datasets.svg import SESYDFloorPlan as CADDataset
    elif opt.graph == 'shape':
        from  Datasets.svg2 import SESYDFloorPlan as CADDataset
    elif opt.graph == 'bezier_edge_attr':
        from  Datasets.svg3 import SESYDFloorPlan as CADDataset
    elif opt.graph == 'bezier_cc':
        from  Datasets.graph_dict import SESYDFloorPlan as CADDataset
    elif opt.graph == 'bezier_cc_bb':
        from  Datasets.graph_dict2 import SESYDFloorPlan as CADDataset
    

    test_dataset = CADDataset(opt.data_dir, opt, partition = opt.phase, data_aug = False, do_mixup = False)
    test_loader = DataLoader(test_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=8, 
        collate_fn = InMemoryDataset.collate)

#    if opt.multi_gpus:
#        train_loader = DataListLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
#    else:
#        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    opt.n_classes = len(list(test_dataset.class_dict.keys()))
    classes = list(range(0, opt.n_classes))
    opt.in_channels = test_dataset[0].x.shape[1]

    logging.info('===> Loading the network ...')
    if opt.arch == 'votenet':
        from votenet import SparseCADGCN, DetectionLoss
    elif opt.arch == 'centernet':
        from architecture import SparseCADGCN, DetectionLoss
    elif opt.arch == 'centernet2':
        from architecture2 import SparseCADGCN, DetectionLoss
    elif opt.arch == 'centernet3cc':
        from architecture3cc import SparseCADGCN, DetectionLoss
    elif opt.arch == 'two_stage':
        from two_stage import SparseCADGCN, DetectionLoss
    elif opt.arch == 'two_stage2':
        from two_stage2 import SparseCADGCN, DetectionLoss
    elif opt.arch == 'cluster':
        from architecture_cluster import SparseCADGCN, DetectionLoss
    elif opt.arch == 'centernet3cc_rpn':
        from architecture3cc_rpn import SparseCADGCN, DetectionLoss

    model = SparseCADGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = DataParallel(SparseDeepGCN(opt)).to(opt.device)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    logging.info(model)

    model.eval()  # Set in evaluation mode

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()

    with torch.no_grad():
        sample_metrics = []
        labels = []
        for i_batch, (data, slices) in enumerate(test_loader):
            pos_slice = slices['pos']
            for key in slices:
                if 'edge' in key:
                    s = slices[key]
                    #print(key, s)
                    o = getattr(data, key)
                    for i_s in range(0, len(s) - 1):
                        start = s[i_s]
                        end = s[i_s + 1]
                        o[start:end] += pos_slice[i_s]
                    setattr(data, key, o)
                elif key == 'bbox_idx':
                    bbox_offset = slices['labels']
                    s = slices[key]
                    #print(key, s)
                    o = getattr(data, key)               
                    for i_s in range(0, len(s) - 1):
                        start = s[i_s]
                        end = s[i_s + 1]
                        o[start:end] += bbox_offset[i_s]
                    setattr(data, key, o)

            if opt.arch == 'centernet':
                pred_cls, pred_coord, pred_coord_max = model.predict(data.x.cuda(), 
                    [data.edge.cuda().T, data.edge_control.cuda().T, data.edge_pos.cuda().T], 
                    [None, None, data.e_weight_pos.cuda().T]
                )
            else:
                #pred_cls, pred_coord, rpn_coord, pred_coord_max = model.predict(data, slices)
                out = model.predict(data, slices)
                pred_cls = out[0]
                pred_coord = out[1]
                pred_coord_max = out[-1]

            if pred_coord_max is not None:
                pred_coord = pred_coord_max
            
            #print(slices)
            image_slice = slices['x']
            label_slice = slices['gt_labels']

            for i in range(0, len(image_slice) - 1):
                start = image_slice[i]
                end = image_slice[i + 1]
                is_control_mask = ~data.is_control[start:end].squeeze()
                pos_img = data.pos[start:end][is_control_mask].cuda()


                if opt.arch != 'centernet3cc_rpn':
                    pred_coord_img = pred_coord[start:end][is_control_mask]
                    pred_cls_img = pred_cls[start:end][is_control_mask]
                    
                else:
                    t_start = slices['bbox'][i]
                    t_end = slices['bbox'][i + 1]
                    
                    pred_coord_img = pred_coord[t_start:t_end]

                    pred_cls_img = pred_cls[t_start:t_end]
                    _, pred_label = pred_cls_img.max(1)
                    label = data.labels[t_start:t_end].cuda()
                    is_bad_case = (label != pred_label)
                    pred_coord_img = pred_coord_img[is_bad_case]
                    pred_cls_img = pred_cls_img[is_bad_case]
                    pred_label = pred_label[is_bad_case]
    
                if opt.arch == 'centernet3cc':
                    not_super = ~data.is_super[start:end][is_control_mask].squeeze()
                    print(pred_cls_img.size(), not_super.size())
                    pred_cls_img = pred_cls_img[not_super]
                    pred_coord_img = pred_coord_img[not_super]
                    pos_img = pos_img[not_super]

                #print('before', pred_coord_img, data.width[i], data.height[i])


                pred_coord_img[:, 0] *= data.width[i]
                pred_coord_img[:, 2] *= data.width[i]
                pred_coord_img[:, 1] *= data.height[i]
                pred_coord_img[:, 3] *= data.height[i]
                pos_img[:, 0] *= data.width[i]
                pos_img[:, 1] *= data.height[i]
                
                #pred_coord_img[:, 0:2] = pos_img - pred_coord_img[:, 0:2]
                #pred_coord_img[:, 2:4] = pos_img + pred_coord_img[:, 2:4]
                if opt.arch != 'centernet3cc_rpn':
                    p0 = pred_coord_img[:, 0:2] - pred_coord_img[:, 2:4] / 2.0
                    p1 = pred_coord_img[:, 0:2] + pred_coord_img[:, 2:4] / 2.0
                    pred_coord_img[:, 0:2] = p0
                    pred_coord_img[:, 2:4] = p1
                
                #pred_coord_img[:, 0:2] = pos_img
                #pred_coord_img[:, 2:4] = pos_img
                print(pred_coord_img.cpu().numpy().shape, np.ones((pred_coord_img.size(0), 1)).shape, pred_label.cpu().numpy()[:, None].shape)
                detections = [np.concatenate([pred_coord_img.cpu().numpy(), 
                    np.ones([pred_coord_img.size(0), 1]), 
                    np.ones([pred_coord_img.size(0), 1]), 
                    pred_label.cpu().numpy()[:, None]], axis = 1)]
                
                # Log progress
                current_time = time.time()
                inference_time = datetime.timedelta(seconds=current_time - prev_time)
                prev_time = current_time
                print("\t+ Batch %d-%d, Inference Time: %s" % (i_batch, i, inference_time))

                # Save image and detections
                imgs.extend([data.filepath[i]])
                img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path.replace('svg', 'tiff')))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                #detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = np.array(list(set(detections[:, -1])), dtype=np.int)
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            
            filename = os.path.dirname(path).split('/')[-1] + '_' + os.path.basename(path).split(".")[0]

            output_path = os.path.join("output", f"{filename}.png")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()
