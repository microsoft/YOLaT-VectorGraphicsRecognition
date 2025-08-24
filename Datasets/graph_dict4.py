# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import os
from rich import print
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from xml.dom.minidom import parse, Node, parseString

from torch_geometric.data import Data
from Datasets.svg_parser import SVGParser
from Datasets.svg_parser import SVGGraphBuilderBezier2 as SVGGraphBuilderBezier
from sklearn.metrics.pairwise import euclidean_distances

#import networkx as nx
import math
import cv2
import random
from scipy.spatial.distance import pdist, squareform

from PIL import Image

from utils.det_util import bbox_iou_ios_cpu, intersect_bb_idx
#from a2c import a2c

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x
    

class idxTree:
    def __init__(self):
        self.children = []
        self.value = {}

class SESYDFloorPlan(torch.utils.data.Dataset):
    def __init__(self, root, opt, 
        partition = 'train', 
        data_aug = False, 
        do_mixup = True, 
        drop_edge = 0, 
        bbox_file_postfix = '_bb.pkl', 
        bbox_sampling_step = 5):
        super(SESYDFloorPlan, self).__init__() 
        
        # svg_list = open(os.path.join(root, partition + '_list.txt')).readlines()
        svg_list = open(os.path.join(root, 'test_length_1.txt')).readlines()
        svg_list = [os.path.join(root, line.strip()) for line in svg_list]
        self.opt = opt
        self.graph_builder = SVGGraphBuilderBezier()
        #print(svg_list)

        self.pos_edge_th = opt.pos_edge_th
        self.data_aug = data_aug
        self.svg_list = svg_list
        self.bbox_sampling_step = bbox_sampling_step
        self.bbox_file_postfix = bbox_file_postfix
        
        stats = pickle.load(open(os.path.join(root, 'stats.pkl'), 'rb'))
        self.attr_mean = np.array([stats['angles']['mean'], stats['distances']['mean']])
        self.attr_std = np.array([stats['angles']['std'], stats['distances']['std']])

        self.normalize_bbox = False
        self.do_mixup = do_mixup

        self.partition = partition
        
        if 'diagram' in os.path.dirname(svg_list[0]):
            self.class_dict = {
                'diode2':0, 
                'capacitor2': 1, 
                'diode3': 2, 
                'earth': 3, 
                'battery1': 4, 
                'battery2': 5, 
                'core-iron': 6, 
                'outlet': 7, 
                'transistor-npn': 8, 
                'capacitor1': 9, 
                'resistor': 10, 
                'relay': 11, 
                'core-air': 12, 
                'transistor-mosfetn': 13, 
                'transistor-mosfetp': 14, 
                'core-hiron': 15, 
                'transistor-pnp': 16, 
                'diode1': 17, 
                'diodephoto': 18, 
                'gate-ampli':19, 
                'unspecified': 20, 
                'None': 21
            }
            
        elif 'vage' in os.path.dirname(svg_list[0]):
            self.class_dict = {
                'mark': 0,
                'legend_symbol': 1,
                'legend_label': 2,
                'legend_title': 3,
                'X-axis_label': 4,
                'X-axis_title': 5,
                'Y-axis_label': 6,
                'Y-axis_title': 7,
                'None': 8
            }

        elif 'svg2video' in os.path.dirname(svg_list[0]): # or "plotly" 
            self.class_dict = {
            'mark_line': 0,
            'mark_point': 1,
            'mark_bar': 2,
            'mark_pie': 3,
            'mark_data': 4,
            'legend_symbol': 5,
            'legend_label': 6,
            'legend_title': 7,
            'X-axis_tick': 8,
            'X-axis_domain': 9,
            'X-axis_label': 10,
            'X-axis_title': 11,
            'Y-axis_tick': 12,
            'Y-axis_domain': 13,
            'Y-axis_label': 14,
            'Y-axis_title': 15,
            'chart_title': 16,
            'None': 17
            }
            
        elif "plotly" in os.path.dirname(svg_list[0]):
            self.class_dict = {
                'mark_line': 0,
                'mark_bar': 1,
                'mark_point': 2,
                'mark_pie': 3,
                'mark_box': 4,
                'mark_violin': 5,
                'mark_data': 6,
                'mark_contourfill': 7,
                'mark_contourline': 8,
                'mark_area': 9,
                'legend_symbol': 10,
                'legend_label': 11,
                'X-axis_label': 12,
                'X-axis_title': 13,
                'Y-axis_label': 14,
                'Y-axis_title': 15,
                'chart_title': 16,
                'sankey_link': 17,
                'sankey_node': 18,
                'sankey_node_label': 19,
                'colorbar': 20,
                'colorbar_label': 21,
                'colorbar_line': 22,
                'None': 23
            }
            
        elif 'Venngage' in os.path.dirname(svg_list[0]):
            self.class_dict = {
            'mark_line': 0,
            'mark_point': 1,
            'mark_bar': 2,
            'mark_pie': 3,
            'mark_data': 4,
            'legend_symbol': 5,
            'legend_label': 6,
            'legend_title': 7,
            'X-axis_tick': 8,
            'X-axis_domain': 9,
            'X-axis_label': 10,
            'X-axis_title': 11,
            'Y-axis_tick': 12,
            'Y-axis_domain': 13,
            'Y-axis_label': 14,
            'Y-axis_title': 15,
            'chart_title': 16,
            'None': 17
        }

        elif "annotation" in os.path.dirname(svg_list[0]):
            self.class_dict = {
                'mark': 0,
                'legend_symbol': 1,
                'legend_label': 2,
                'legend_title': 3,
                'X-axis_label': 4,
                'X-axis_title': 5,
                'Y-axis_label': 6,
                'Y-axis_title': 7,
                'mark_data': 8,
                'annotation_line': 9,
                'annotation_arrow': 10,
                'annotation_circle': 11,
                'annotation_band': 12,
                'annotation_text':13,
                'chart_title': 14,
                'None': 15
            }

        else:
            print('Floorplan')
            self.class_dict = {
                'armchair':0, 
                'bed':1, 
                'door1':2, 
                'door2':3, 
                'sink1':4, 
                'sink2':5, 
                'sink3':6, 
                'sink4':7, 
                'sofa1':8, 
                'sofa2':9, 
                'table1':10, 
                'table2':11, 
                'table3':12, 
                'tub':13, 
                'window1':14, 
                'window2':15, 
                'None': 16
            }

        self.n_classes = len(list(self.class_dict.keys()))
        print(self.class_dict)
        
        self.n_objects = 13238

    def __len__(self):
        return len(self.svg_list)
        
    def get_anchor(self):
        bboxes = [[] for i in range(len(list(self.class_dict.keys())))]
        for filepath in self.svg_list:
            p = SVGParser(filepath)
            width, height = p.get_image_size()
            gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
            whs = gt_bbox[:, 2:] -  gt_bbox[:, 0:2]
            for wh, l in zip(whs, gt_labels):
                print(l)
                bboxes[l].append(wh)
            
        bboxes = np.array(bboxes)
        for wh in bboxes:
            mean_box = np.median(wh, 0)
            print(mean_box, np.mean(wh, 0), np.max(wh, 0), np.min(wh, 0))
        print(bboxes.shape)
        raise SystemExit

    def _get_bbox(self, path, width, height):
        ### Wrong for selecting wrong xml for svg file
        try:
            if self.partition == "detect":
                dom = parse('/home/v-sdou/teamdrive/msrashaiteamdrive/users/sdou/YOLaT-VectorGraphicsRecognition/data/svg2video/line_vega_datasets_3k/plain_1.xml')
            else:
                dom = parse(path.replace('.svg', '.xml'))
        except:
            print(self.partition)
            print("No xml file:", path)
            raise SystemExit
        # dom = parse(path.replace('.svg', '.xml'))
        root = dom.documentElement

        

        nodes = []
        for tagname in ['a', 'o']:
            nodes += root.getElementsByTagName(tagname)
        
        bbox = []
        labels = []
        for node in nodes:
            for n in node.childNodes:
                if n.nodeType != Node.ELEMENT_NODE:
                    continue
                x0 = float(n.getAttribute('x0')) / width
                y0 = float(n.getAttribute('y0')) / height
                x1 = float(n.getAttribute('x1')) / width 
                y1 = float(n.getAttribute('y1')) / height

                label = n.getAttribute('label')
                bbox.append((x0, y0, x1, y1))
                labels.append(self.class_dict[label])

        return np.array(bbox), np.array(labels)

    def refine_gt(self, graph_dict, bbox):
        pos = graph_dict['pos']['spatial']
        is_control = graph_dict['attr']['is_control']
        #print(pos.shape, bbox.shape, labels.shape)
        #print(np.max(pos[:, 0]), np.max(pos[:, 1]))

        th = 1e-3
        gt_bb = []
        gt_cls = []
        gt_object = []

        for node_idx, p in enumerate(pos):
            if is_control[node_idx]: 
                gt_bb.append((0, 0, 0, 0))
                gt_cls.append((0))
                gt_object.append((0))
                continue

            diff_0 = p[None, :] - bbox[:, 0:2]
            diff_1 = p[None, :] - bbox[:, 2:]
            in_object = (diff_0[:, 0] >= -th) & (diff_0[:, 1] >= -th) & (diff_1[:, 0] <= th) & (diff_1[:, 1] <= th)
            
            object_index = np.where(in_object)[0]
            if len(object_index) > 1:
                #print(object_index)
                #print('node', p[0] * width, p[1] * height, 'is inside more than one object')
                candidates = bbox[object_index]
                s = euclidean_distances(p[None, :], candidates[:, 0:2])[0]
                #print(np.argsort(s))
                object_index = object_index[np.argsort(s)]
                #print(candidates, s, object_index)
            elif len(object_index) == 0:
                #print(diff_0 * [width, height], diff_1* [width, height])
                #print(object_index)
                print('node', p[0] * width, p[1] * height, 'outside all object')
                #for i, line in enumerate(bbox[:, 0:2] * [width, height]):
                #    print(i, line)
                raise SystemExit
            cls = labels[object_index[0]]
            bb = bbox[object_index[0]]
            '''
            h = bb[3] - bb[1]
            w = bb[2] - bb[0]
            offset_x = bb[0] - p[0]
            offset_y = bb[1] - p[1]
            gt_bb.append((offset_x, offset_y, w, h))
            '''
            gt_bb.append(bb)
            gt_cls.append(cls)
            gt_object.append(object_index[0])
        
        #assign label to control
        control_neighboor = {}
        for e in graph_dict['edge']['control']:
            #print(is_control[e[0]], is_control[e[1]])
            
            if not is_control[e[0]] and is_control[e[1]]:
                c_node = e[1]
                node = e[0]
            elif not is_control[e[1]] and is_control[e[0]]:
                c_node = e[0]
                node = e[1]
            else:
                continue

            if c_node not in control_neighboor:
                control_neighboor[c_node] = []
            control_neighboor[c_node].append(node)
        #print(graph_dict['edge']['control'])
        #print(control_neighboor)
        
        #print(gt_bb, gt_cls)
        for node_idx, p in enumerate(pos):
            if is_control[node_idx]: 
                #print(control_neighboor[node_idx][0])
                gt_bb[node_idx] = gt_bb[control_neighboor[node_idx][0]]
                gt_cls[node_idx] = gt_cls[control_neighboor[node_idx][0]]
                gt_object[node_idx] = gt_object[control_neighboor[node_idx][0]]
                #raise SystemExit
        #print(gt_bb, gt_cls)
        
        return np.array(gt_bb), np.array(gt_cls), np.array(gt_object)

    def __transform__(self, pos, scale, angle, translate):
        scale_m = np.eye(2)
        scale_m[0, 0] = scale
        scale_m[1, 1] = scale

        rot_m = np.eye(2)
        rot_m[0, 0:2] = [np.cos(angle), np.sin(angle)]
        rot_m[1, 0:2] = [-np.sin(angle), np.cos(angle)]

        #print(pos.shape, scale_m[0:2].shape)
        
        #print(pos.shape)
        center = np.array((0.5, 0.5))[None, :]
        pos -= center
        if random.choice([True, False]):
            pos[:, 0] = -pos[:, 0]
        if random.choice([True, False]):
            pos[:, 1] = -pos[:, 1]
        pos = np.matmul(pos, rot_m[0:2])
        pos += center
        pos += np.array(translate)[None, :]
        pos = np.matmul(pos, scale_m[0:2])
        return pos

    def __transform_bbox__(self, bbox, scale, angle, translate):
        p0 = bbox[:, 0:2]
        p2 = bbox[:, 2:]
        p1 = np.concatenate([p2[:, 0][:, None], p0[:, 1][:, None]], axis = 1)
        p3 = np.concatenate([p0[:, 0][:, None], p2[:, 1][:, None]], axis = 1)
        
        p0 = self.__transform__(p0, scale, angle, translate)
        p1 = self.__transform__(p1, scale, angle, translate)
        p2 = self.__transform__(p2, scale, angle, translate)
        p3 = self.__transform__(p3, scale, angle, translate)

        
        def bound_rect(p0, p1, p2, p3):
            x = np.concatenate((p0[:, 0][:, None], p1[:, 0][:, None], p2[:, 0][:, None], p3[:, 0][:, None]), axis = 1)
            y = np.concatenate((p0[:, 1][:, None], p1[:, 1][:, None], p2[:, 1][:, None], p3[:, 1][:, None]), axis = 1)
            x_min = x.min(1, keepdims = True)
            x_max = x.max(1, keepdims = True)
            y_min = y.min(1, keepdims = True)
            y_max = y.max(1, keepdims = True)

            return np.concatenate([x_min, y_min, x_max, y_max], axis = 1)
        return bound_rect(p0, p1, p2, p3)

    def random_transfer(self, pos, bbox, gt_bbox, bbox_targets):
        scale_ratio = 0.6
        scale = (np.random.random() * 2 - 1) * scale_ratio + 1 #np.random.random() * 0.2 + 0.9
        angle = np.random.random() * np.pi * 2

        translate_ratio = 0.1
        translate = [0, 0]
        translate[0] = (np.random.random() * 2 - 1) * translate_ratio #np.random.random() * 0.2 - 0.1
        translate[1] = (np.random.random() * 2 - 1) * translate_ratio #np.random.random() * 0.2 - 0.1

        pos = self.__transform__(pos, scale, angle, translate)
        #bbox = self.__transform_bbox__(bbox, scale, angle, translate)
        gt_bbox = self.__transform_bbox__(gt_bbox, scale, angle, translate)
        bbox_targets = self.__transform_bbox__(bbox_targets, scale, angle, translate)

        return pos, bbox, gt_bbox, bbox_targets

    def getEdgeWeight(self, pos, edge):
        distance = euclidean_distances(pos, pos)
        w = 1 / np.exp(distance)
        weight = []

        for e in edge:
            weight.append(w[e[0], e[1]])
        return np.array(weight)

    def _get_proposal(self, graph_dict, gt_bbox, gt_labels, bbox_sampling_step = -1, vis_print = True, for_sub_cluster=False):
        vis_print = False
        cc = graph_dict['cc']
        if vis_print: print(len(cc), cc)
        # print("cluster:", len(cc), cc)
        pos = graph_dict['pos']['spatial']

        edge = graph_dict['edge']['shape']
        # edge_element = graph_dict['edge']['element']
        # print(edge_element)
        edge_super = graph_dict['edge']['super']
        e_attr = graph_dict['edge_attr']['shape']
        e_attr_super = graph_dict['edge_attr']['super']
        # e_attr_element = graph_dict['edge_attr']['element']

        is_super = graph_dict['attr']['is_super']
        is_control = graph_dict['attr']['is_control']
        feats = np.concatenate((graph_dict['attr']['color'], graph_dict['attr']['type'], graph_dict['attr']['stroke_width']), axis=1)

        values = graph_dict['attr']['value']

        width = graph_dict['img_width']
        height = graph_dict['img_height']

               
        o2n = {}
        count = 0
        for i, ic in enumerate(is_control):
            if not ic:
                o2n[i] = count
                count += 1

        new_edge = []
        for e in edge:
            new_edge.append([o2n[e[0]], o2n[e[1]]])
        edge = np.array(new_edge)

        new_cc = []
        for cluster in cc:
            new_cluster = []
            for idx in cluster:
                new_cluster.append(o2n[idx])
            new_cc.append(new_cluster)
        cc = new_cc

        if vis_print: print("new_cc:", cc, "new_cc shape:", len(cc))

        new_edge = []
        for e in edge_super:
            new_edge.append([o2n[e[0]], o2n[e[1]]])
        edge_super = np.array(new_edge)

        # new_edge = []
        # for e in edge_element:
        #     new_edge.append([o2n[e[0]], o2n[e[1]]])
        # edge_element = np.array(new_edge)

        not_control = (is_control == 0)[:, 0]
        pos = pos[not_control]
        feats = feats[not_control]
        is_super = is_super[not_control]
        
        if self.do_mixup:
            cc, pos, edge, edge_super, e_attr, e_attr_super, is_super = self.mixup(cc, pos, edge, edge_super, e_attr, e_attr_super, is_super)

        new_pos = []
        new_edge = []
        new_feats = []

        # add scene graph
        scene_spatial = []
        scene_edge = []
        scene_edge_element = []
        scene_feats = []

        new_values = []
        new_edge_super = []
        # new_edge_element = []
        new_e_attr = []
        new_e_attr_super = []
        # new_e_attr_element = []
        new_is_super = []
        new_labels = []

        new_bbox = []
        bbox_targets = []
        bbox_idx = []
        stat_feats = []
        has_objs = []
        offset = 0
        roots = []
        bbox_count = 0

        subcluster_slice_pos = [0]
        subcluster_slice_edge = [0]
        subcluster_slice_super = [0]
        # subcluster_slice_element = [0]
        subcluster_slice_bbox = [0]

        for cc_idx, cluster in enumerate(cc):
            #cluster = [i for i in cluster if not is_super[i]]

            pos_cluster = pos[cluster, :]
            values_cluster = [values[i] for i in graph_dict['cc'][cc_idx]]
            feats_cluster = feats[cluster, :]
            if vis_print: print('cc_idx:', cc_idx,'pos_cluster:', pos_cluster)
            if any(np.array(pos_cluster.shape) == 0): continue

            max_x = pos_cluster[:, 0].max(0)
            min_x = pos_cluster[:, 0].min(0)
            max_y = pos_cluster[:, 1].max(0)
            min_y = pos_cluster[:, 1].min(0)

            #########################
            x_values = sorted(pos_cluster[:, 0])
            y_values = sorted(pos_cluster[:, 1])
            def merge_values(values):
                new_values  = [values[0]]
                for i in range(1, len(values)):
                    if values[i] != values[i - 1]: #> 1e-3:
                        new_values.append(values[i])
                return new_values
            x_values = merge_values(x_values)
            y_values = merge_values(y_values)
            # print(x_values, y_values)

            def get_values_dict(values):
                values_dict = {}
                for i, v in enumerate(values):
                    values_dict[v] = i
                return values_dict
            x_values_dict = get_values_dict(x_values)
            y_values_dict = get_values_dict(y_values)

            use_bit = False

            #point_exist = np.ones((len(y_values), len(x_values))).astype(np.int8) * (-1)
            point_exist = [[[] for j in range(len(x_values))] for i in range(len(y_values))]
            #print(x_values, y_values)

            pos_idx = range(pos_cluster.shape[0])
            if use_bit and len(pos_idx) > 64:
                print('more than 64 points in cc', len(pos_idx))
                pos_idx = random.sample(pos_idx, 64)

            for i in pos_idx:
                p = pos_cluster[i]                
                point_exist[y_values_dict[p[1]]][x_values_dict[p[0]]].append(i)
            
            def set_bit(value, bit):
                return value | (1<<bit)

            d00 = [[None for i in range(len(x_values))] for j in range(len(y_values))]
            d00[0][0] = point_exist[0][0]

            for i in range(1, len(x_values)):
                d00[0][i] = d00[0][i - 1] + point_exist[0][i]

            for i in range(1, len(y_values)):
                d00[i][0] = d00[i - 1][0] + point_exist[i][0]

            d_row = [[None for i in range(len(x_values))] for j in range(len(y_values))]
            for i in range(0, len(x_values)):
                d_row[0][i] = d00[0][i]

            for i in range(1, len(y_values)):
                d_row[i][0] = point_exist[i][0]
 
            for y in range(1, len(y_values)):
                for x in range(1, len(x_values)):
                    d_row[y][x] = d_row[y][x - 1] + point_exist[y][x]
                    d00[y][x] = d00[y - 1][x] + d_row[y][x]
            
            for y in range(0, len(y_values)):
                for x in range(0, len(x_values)):
                    d00[y][x] = set(d00[y][x])

            if for_sub_cluster:
                sub_clusters = []
                # print(len(y_values),len(x_values))

                x_step = (max_x - min_x) / bbox_sampling_step #10 #5 #5#25
                y_step = (max_y - min_y) / bbox_sampling_step #10 #5 #5
                #print(d00)

                if vis_print: print("X:", min_x, max_x, x_step)
                if vis_print: print("Y:", min_y, max_y, y_step)
                ### x_step may be zero
                if x_step == 0:
                    x_grids = np.array([min_x])
                else:
                    x_grids = np.arange(min_x, max_x, x_step)
                    x_grids = np.append(x_grids, max_x)
                if y_step == 0:
                    y_grids = np.array([min_y])
                else:
                    y_grids = np.arange(min_y, max_y, y_step)
                    y_grids = np.append(y_grids, max_y)

                # print(x_grids, y_grids)
                def move_endpoint(x, values, bound):
                    if x >= len(values):
                        return x - 1

                    while values[x] <= bound:
                        x += 1
                        if x >= len(values):
                            break
                    return x - 1

                def move_endpoint_close(x, values, bound):
                    if x >= len(values):
                        return x - 1

                    while values[x] < bound:
                        x += 1
                        if x >= len(values):
                            break
                    return x - 1
                
                prev_y0 = -1
                grid_y0 = 0
                for i_grid_y0, grid_y0 in enumerate(y_grids):
                    y0 = move_endpoint_close(prev_y0 + 1, y_values, grid_y0)
                    if y0 != len(y_values): y0 += 1
                    if y0 == prev_y0: continue
                    prev_y0 = y0
                    
                    grid_x0 = x_values[0]
                    prev_x0 = -1
                    for i_grid_x0, grid_x0 in enumerate(x_grids):
                        x0 = move_endpoint_close(prev_x0 + 1, x_values, grid_x0)
                        if x0 != len(y_values): x0 += 1
                        if x0 == prev_x0: continue
                        prev_x0 = x0
                        
                        #grid_y1 = grid_y0
                        prev_y1 = y0
                        for grid_y1 in y_grids[i_grid_y0 + 1 :]:
                            y1 = move_endpoint(prev_y1 + 1, y_values, grid_y1)
                            #if prev_y1 + 1 < len(y_values):
                            #    print(prev_y1 + 1, y_values[prev_y1 + 1], 'to', grid_y1)
                            if y1 == prev_y1: continue
                            #print('---------------', prev_y1, 'to', y1, y_values[prev_y1], 'to', grid_y1)
                            prev_y1 = y1
                            
                            #grid_x1 = grid_x0
                            prev_x1 = x0
                            for grid_x1 in x_grids[i_grid_x0 + 1:]:
                                x1 = move_endpoint(prev_x1 + 1, x_values, grid_x1)
                                if x1 == prev_x1: continue
                                prev_x1 = x1
                                
                                if use_bit:
                                    if x0 > 0 and y0 > 0:
                                        dd = d00[y1][x1] - (d00[y1][x0 - 1] | d00[y0 - 1][x1])
                                    elif x0 > 0 and y0 == 0:
                                        dd = d00[y1][x1] - d00[y1][x0 - 1]
                                    elif y0 > 0 and x0 == 0:
                                        dd = d00[y1][x1] - d00[y0 - 1][x1]
                                    else:
                                        dd = d00[y1][x1]
                                    if dd == 0: continue
                                    count = 0
                                    sub_c = []
                                    while dd != 0:
                                        if dd & 1:
                                            sub_c.append(count)
                                        count += 1
                                        dd = dd >> 1
                                    sub_c = [cluster[ii] for ii in sub_c]
                                    sub_clusters.append(tuple(sub_c))
                                else:
                                    if x0 > 0 and y0 > 0:
                                        dd = d00[y1][x1].difference(d00[y1][x0 - 1]).difference(d00[y0 - 1][x1])
                                    elif x0 > 0 and y0 == 0:
                                        dd = d00[y1][x1].difference(d00[y1][x0 - 1])
                                    elif y0 > 0 and x0 == 0:
                                        dd = d00[y1][x1].difference(d00[y0 - 1][x1])
                                    else:
                                        dd = d00[y1][x1]
                                    #print(x0, y0, x1, y1, 'fooo')
                                    sub_c = [cluster[ii] for ii in dd]
                                    sub_clusters.append(tuple(sorted(sub_c)))
                
                sub_clusters = list(set(sub_clusters))
                # print("sub clusters:", sub_clusters)
                # raise SystemError
            else:
                sub_clusters = [cluster]
            
            #########################
            def get_adj(edge):
                #adj = -np.ones((pos.shape[0], pos.shape[0])).astype(np.int)
                adj = [[[] for j in range(pos.shape[0])] for j in range(pos.shape[0])]
                for i, e in enumerate(edge):
                    #adj[e[0], e[1]] = i
                    #adj[e[1], e[0]] = i
                    adj[e[0]][e[1]].append(i)
                    adj[e[1]][e[0]].append(i)
                return adj
            
            A = get_adj(edge)
            A_super = get_adj(edge_super)
            # A_element = get_adj(edge_element)
            #print(A)

            bbox_cc = np.array([min_x, min_y, max_x, max_y])[None, :]
            if vis_print: print('bbox_cc:', bbox_cc)
            # if vis_print: print('gt_bbox:', gt_bbox)

            gt_bbox_idx_valid = intersect_bb_idx(bbox_cc, gt_bbox)
            
            # print("cluster:", cluster)
            # print('bbox_cc:', bbox_cc)
            # print('gt_bbox:', gt_bbox[gt_bbox_idx_valid]) # 
            # print("gt_bbox_idx_valid & label", gt_bbox_idx_valid, gt_labels[gt_bbox_idx_valid])

            if gt_bbox_idx_valid.shape[0] == 0:
                # raise SystemExit
                if self.partition == "detect":
                    pass
                else:
                    if vis_print: print('cc has no intersect gt bbox')
                    continue

            sub_bbox_n = 0
            for idxs in sub_clusters:
                o2n = {}
                for i, idx in enumerate(idxs):
                    o2n[idx] = i
                pos_bbox = pos[idxs, :]

                # value_bbox = [values_cluster[i] for i in idxs]
                is_super_bbox = is_super[idxs, :]
                #idxs = set(idxs)
                # print(idxs, values_cluster)
                
                edge_idxs = []
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        #if A[idxs[i], idxs[j]] >= 0:
                            #edge_idxs.append(A[idxs[i], idxs[j]])
                        edge_idxs+= A[idxs[i]][idxs[j]]

                edge_bbox = edge[edge_idxs]
                if vis_print: print('edge_bbox:', edge_bbox)
                if edge_bbox.shape[0] == 0:
                    continue

                edge_bbox = np.array([[o2n[e[0]] + offset, o2n[e[1]] + offset] for e in edge_bbox])
                e_attr_bbox = e_attr[edge_idxs]
                
                # edge_idxs = []
                # for i in range(len(idxs)):
                #     for j in range(i + 1, len(idxs)):
                #         #if A[idxs[i], idxs[j]] >= 0:
                #             #edge_idxs.append(A[idxs[i], idxs[j]])
                #         edge_idxs+= A_element[idxs[i]][idxs[j]]

                # edge_element_bbox = edge_element[edge_idxs]
                # edge_element_bbox = np.array([[o2n[e[0]] + offset, o2n[e[1]] + offset] for e in edge_element_bbox])
                # e_attr_element_bbox = e_attr_element[edge_idxs]
                
                edge_idxs = []
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        #if A_super[idxs[i], idxs[j]] >= 0:
                            #edge_idxs.append(A_super[idxs[i], idxs[j]])
                        edge_idxs += A_super[idxs[i]][idxs[j]]
                
                edge_super_bbox = edge_super[edge_idxs]
                edge_super_bbox = np.array([[o2n[e[0]] + offset, o2n[e[1]] + offset] for e in edge_super_bbox])
                e_attr_super_bbox = e_attr_super[edge_idxs]
                
                # print(count, offset, pos_bbox.shape, is_super_bbox.shape, edge_bbox.shape, edge_super_bbox.shape, e_attr_bbox.shape, e_attr_super_bbox.shape)

                max_x = pos_bbox[:, 0].max(0)
                min_x = pos_bbox[:, 0].min(0)
                max_y = pos_bbox[:, 1].max(0)
                min_y = pos_bbox[:, 1].min(0)

                # the operation will skip tick and domain so I annotate it 
                # if max_x - min_x < 1e-4 or max_y - min_y < 1e-4:
                #     continue

                proposal = np.array([min_x, min_y, max_x, max_y])[None, :]
                if self.partition == "detect":
                    label = self.n_classes - 1
                    bbox_target = np.zeros((1, 4))
                    has_obj = 1
                
                else:
                    iou, ios = bbox_iou_ios_cpu(proposal, gt_bbox[gt_bbox_idx_valid, :])
                    idx_gt = np.argmax(iou)
                    
                    if vis_print: print('proposal, iou:', proposal, iou)
                    
                    ### 
                    # print('proposal:', proposal)
                    # print('IOU:', iou)

                    # for vega-lite, 0.7 may best
                    # However for plotly, 0.7 will get wrong label, such as mark point->none. We chose 0.57 as theahold
                    theahold = 0.55
                    # add rule to reassign labels
                    
                    # if gt_labels[gt_bbox_idx_valid[idx_gt]] in [8, 9, 12, 13]:
                    #     label = gt_labels[gt_bbox_idx_valid[idx_gt]]                    
                    #     bbox_target = gt_bbox[gt_bbox_idx_valid[idx_gt]][None, :]

                    # rect may overlay with area, cause high IOU to generate wrong label
                    # remove rect
                    if iou[idx_gt] > theahold and feats_cluster[0,9]!=1:
                        label = gt_labels[gt_bbox_idx_valid[idx_gt]]                    
                        bbox_target = gt_bbox[gt_bbox_idx_valid[idx_gt]][None, :]
                    else:
                        label = self.n_classes - 1
                        bbox_target = np.zeros((1, 4))

                    if vis_print: print("label & bbox_target:", label, bbox_target)

                    # label = self.n_classes - 1
                    # bbox_target = np.zeros((1, 4))
                    # if vis_print: print("label & bbox_target:", label, bbox_target)

                    for idx in gt_bbox_idx_valid:
                        gt_bb = gt_bbox[idx]
                        if np.allclose(gt_bb, proposal, rtol=0, atol=1e-8):
                            label = gt_labels[idx]
                            bbox_target = gt_bbox[idx][None, :]

                        # elif :
                        #     label = gt_labels[idx]
                        #     bbox_target = gt_bbox[idx][None, :]

                    if vis_print: print("label & bbox_target:", label, bbox_target)

                    idx_gt = np.argmax(iou)
                    if ios[idx_gt] > theahold:
                        has_obj = 1
                    else:
                        has_obj = 0
                
                ######################obtain stats#################################
                stats = []
                n_points = pos_bbox.shape[0]
                n_edges = edge_bbox.shape[0]
                
                n_angle_less90 = 0
                n_angle_90 = 0
                n_angle_more90 = 0

                # because the index out of the range, we extend the range
                adj = [set() for i in range(int(pos.shape[0]*1.5))]
                # print(pos.shape[0])

                for e in edge_bbox:
                    # print(e)
                    adj[e[0] - offset].add(e[1] - offset)
                    adj[e[1] - offset].add(e[0] - offset)
                    
                angles = []
                for anchor, neighbors in enumerate(adj):
                    neighbors = list(neighbors)
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            p0 = pos_bbox[neighbors[i]]
                            p1 = pos_bbox[neighbors[j]]
                            p_anchor = pos_bbox[anchor]
                            v0 = p0 - p_anchor
                            v1 = p1 - p_anchor

                            dot = v0[0] * v1[0] + v0[1] * v1[1]
                            if dot <= -1e-2:
                                n_angle_more90 += 1
                            elif dot >= 1e-2:
                                n_angle_less90 += 1
                            elif np.abs(dot) < 1e-2:
                                n_angle_90 +=1
                            angles.append(dot)
                            
                width = max_x - min_x
                height = max_y - min_y

                # the operation will skip tick and domain
                # if len(angles) == 0:
                #     print('angles')
                #     continue
                if len(angles) == 0:
                    angles = [1]
                angles = np.array(angles)
                mean_angle = np.mean(angles)
                max_angle = np.max(angles)
                min_angle = np.min(angles)
                std_angle = np.std(angles)

                long_short_ratio = max(width, height) * 1.0 / min(width, height)

                mean_edge_distance = np.mean(e_attr_bbox[:, -1])
                std_edge_distance = np.std(e_attr_bbox[:, -1])
                mean_edge_angle = np.mean(e_attr_bbox[:, -2])
                std_edge_angle = np.std(e_attr_bbox[:, -2])

                '''
                stat_feat = np.array([n_points, n_edges, n_angle_90, n_angle_less90, n_angle_more90, 
                    width, height, mean_angle, max_angle, min_angle, std_angle, mean_edge_angle, 
                    std_edge_angle, mean_edge_distance, std_edge_distance])[None, :]
                    #, long_short_ratio])[None, :]
                '''
                
                stat_feat = np.array([n_points, n_edges, n_angle_90, n_angle_less90, n_angle_more90, 
                    width, height, mean_angle, max_angle, min_angle, std_angle, mean_edge_distance, std_edge_distance])[None, :]

                if self.normalize_bbox:
                    '''
                    if max_x - min_x >  max_y - min_y:
                        pos_bbox = (pos_bbox - [min_x, min_y]) / [max_x - min_x, max_x - min_x]
                    else:
                        pos_bbox = (pos_bbox - [min_x, min_y]) / [max_y - min_y, max_y - min_y]
                    '''
                    pos_bbox = (pos_bbox - [min_x, min_y]) / [max_x - min_x, max_y - min_y]
                

                subcluster_slice_pos.append(subcluster_slice_pos[-1] + pos_bbox.shape[0])
                subcluster_slice_edge.append(subcluster_slice_edge[-1] + edge_bbox.shape[0])
                subcluster_slice_super.append(subcluster_slice_super[-1] + edge_super_bbox.shape[0])
                # subcluster_slice_element.append(subcluster_slice_element[-1] + edge_element_bbox.shape[0])
                subcluster_slice_bbox.append(subcluster_slice_bbox[-1] + 1)

                new_pos.append(pos_bbox)
                new_feats.append(feats_cluster)
                
                if width==0:
                    h_w = 10
                else:
                    h_w = height/width

                spatial = np.array([min_x, max_x, min_y, max_y, width*height, h_w])
                scene_feats.append(feats_cluster[0,:])
                scene_spatial.append(spatial)

                new_values.extend(values_cluster)
                new_is_super.append(is_super_bbox)
                if edge_bbox.shape[0] > 0:
                    new_edge.append(edge_bbox)
                # if edge_element_bbox.shape[0] > 0:
                #     new_edge_element.append(edge_element_bbox)
                if edge_super_bbox.shape[0] > 0:
                    new_edge_super.append(edge_super_bbox)
                new_e_attr.append(e_attr_bbox)
                new_e_attr_super.append(e_attr_super_bbox)
                # new_e_attr_element.append(e_attr_element_bbox)

                new_labels.append(label)
                has_objs.append(has_obj)
                bbox_idx += [bbox_count] * pos_bbox.shape[0]
                offset += pos_bbox.shape[0]
                new_bbox.append([min_x, min_y, max_x, max_y])
                bbox_targets.append(bbox_target)
                stat_feats.append(stat_feat)

                sub_bbox_n += 1
                bbox_count += 1

            # if vis_print: print(sub_bbox_n, subcluster_slice_pos, subcluster_slice_edge, subcluster_slice_super, subcluster_slice_bbox)
            idx_offset = len(subcluster_slice_bbox) - sub_bbox_n - 1
            # if vis_print: print("new box:", new_bbox)
            sub_bbox = np.array(new_bbox)[subcluster_slice_bbox[idx_offset]:]
            # print(sub_bbox)
            # the operation will skip tick and domain
            # if sub_bbox.size==0: 
            #     print('sub_bbox size')
            #     continue
            # print(sub_bbox, sub_bbox.shape)
            try:
                area = (sub_bbox[:, 2] - sub_bbox[:, 0]) * (sub_bbox[:, 3] - sub_bbox[:, 1])
                #print(area)
                max_idx = np.argmax(area)
            except:
                continue
            #print('root idx', max_idx)
            root = idxTree()
            root.value['idx_pos'] = (subcluster_slice_pos[idx_offset + max_idx], subcluster_slice_pos[idx_offset + max_idx + 1])
            root.value['idx_edge'] = (subcluster_slice_edge[idx_offset + max_idx], subcluster_slice_edge[idx_offset + max_idx + 1])
            root.value['idx_edge_super'] = (subcluster_slice_super[idx_offset + max_idx], subcluster_slice_super[idx_offset + max_idx + 1])
            # root.value['idx_edge_element'] = (subcluster_slice_element[idx_offset + max_idx], subcluster_slice_element[idx_offset + max_idx + 1])
            root.value['idx_bbox'] = subcluster_slice_bbox[idx_offset + max_idx]

            #print(root.value)

            #print(subcluster_slice_pos, len(bbox_idx))
            for i in range(sub_bbox.shape[0]):
                if i == max_idx: continue
                p = idxTree()
                p.value['idx_pos'] = (subcluster_slice_pos[idx_offset + i], subcluster_slice_pos[idx_offset + i + 1])
                p.value['idx_edge'] = (subcluster_slice_edge[idx_offset + i], subcluster_slice_edge[idx_offset + i + 1])
                p.value['idx_edge_super'] = (subcluster_slice_super[idx_offset + i], subcluster_slice_super[idx_offset + i + 1])
                # p.value['idx_edge_element'] = (subcluster_slice_element[idx_offset + i], subcluster_slice_element[idx_offset + i + 1])
                p.value['idx_bbox'] = subcluster_slice_bbox[idx_offset + i]
                root.children.append(p)
            #print(subcluster_slice_pos, subcluster_slice_edge, subcluster_slice_super, subcluster_slice_bbox)
            roots.append(root)
            
            #print(len(bbox_idx), np.concatenate(new_pos, axis = 0).shape)
            # raise SystemExit

        pos = np.concatenate(new_pos, axis = 0)
        feats = np.concatenate(new_feats, axis = 0)
        values = new_values

        # construct the scene graph
        scene_feats = np.stack(scene_feats, axis = 0)
        scene_spatial = np.stack(scene_spatial, axis = 0)
        # print("scene graph:",scene_feats.shape, scene_spatial.shape)

        points = scene_spatial[:, [0, 2]]
        dist_matrix =  squareform(pdist(points))

        spatial_edge = []
        type_edge =  []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if dist_matrix[i,j] < 0.15:
                    spatial_edge.append(np.array([i, j]))
                if dist_matrix[i,j] < 0.15 and scene_feats[i, 6:10].all() == scene_feats[j, 6:10].all():
                    type_edge.append(np.array([i, j]))
            

        spatial_edge = np.concatenate([spatial_edge], axis = 0)
        type_edge = np.concatenate([type_edge], axis = 0)

        # raise SystemExit
        is_super = np.concatenate(new_is_super, axis = 0)
        edge = np.concatenate(new_edge, axis = 0)
        edge_super = np.concatenate(new_edge_super, axis = 0)
        e_attr = np.concatenate(new_e_attr, axis = 0)
        e_attr_super = np.concatenate(new_e_attr_super, axis = 0)
        labels = new_labels
        new_bbox = np.array(new_bbox)
        bbox_targets = np.concatenate(bbox_targets, axis = 0)
        bbox_idx = np.array(bbox_idx)
        is_control = np.zeros((pos.shape[0], 1))
        stat_feats = np.concatenate(stat_feats, axis = 0)
        has_obj = has_objs
        
        e_attr_element = None
        edge_element = None
        return scene_feats, scene_spatial, spatial_edge, type_edge, pos, feats, values, is_super, is_control, \
            edge, edge_super, edge_element, e_attr, e_attr_super, e_attr_element, \
             labels, bbox_idx, new_bbox, bbox_targets, stat_feats, has_obj, roots

    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filepath = self.svg_list[idx]
        
        graph_dict = pickle.load(open(filepath.replace('.svg', '.pkl'), 'rb'))

        try:
            width, height = graph_dict['img_width'], graph_dict['img_height']
        except:
            print(filepath)
            # width, height = 800, 400
        
        gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
        # print(gt_bbox, gt_labels)


        # filename_bbox = filepath.replace('.svg', self.bbox_file_postfix)
        filename_bbox = filepath.replace('.svg', '_GaL.pkl')
        # filename_bbox = filepath.replace('.svg', '_split.pkl')
        preload = True
        if preload:            
            try:
                scene_feats, scene_spatial, spatial_edge, type_edge, pos, feats, value, is_super, is_control, edge, edge_super, edge_element, e_attr, e_attr_super, e_attr_element, \
                    labels, bbox_idx, bbox, bbox_targets,stat_feats, has_obj, roots = pickle.load(open(filename_bbox, 'rb'))
            except:
                #if not os.path.exists(filename_bbox):
                print("Preload is TURE:", filename_bbox)
                scene_feats, scene_spatial, spatial_edge, type_edge, pos, feats,  value, is_super, is_control, edge, edge_super, edge_element, e_attr, e_attr_super, e_attr_element, \
                    labels, bbox_idx, bbox, bbox_targets, stat_feats, has_obj, roots = self._get_proposal(graph_dict, gt_bbox, gt_labels, bbox_sampling_step = self.bbox_sampling_step)
                pickle.dump([scene_feats, scene_spatial, spatial_edge, type_edge, pos, feats, value, is_super, is_control, edge, edge_super, edge_element, e_attr, e_attr_super, e_attr_element,  \
                    labels, bbox_idx, bbox, bbox_targets, stat_feats, has_obj, roots], open(filename_bbox, 'wb'))
        else:
            print("Preload is False:",filename_bbox)
            scene_feats, scene_spatial, spatial_edge, type_edge, pos, feats, value, is_super, is_control, edge, edge_super, edge_element, e_attr, e_attr_super, e_attr_element,  \
                labels, bbox_idx, bbox, bbox_targets, stat_feats, has_obj, roots = self._get_proposal(graph_dict, gt_bbox, gt_labels, bbox_sampling_step = self.bbox_sampling_step)
            # pickle.dump([scene_feats, scene_spatial, spatial_edge, type_edge, pos, feats, value, is_super, is_control, edge, edge_super, edge_element, e_attr, e_attr_super, e_attr_element,  \
            #     labels, bbox_idx, bbox, bbox_targets, stat_feats, has_obj, roots], open(filename_bbox, 'wb'))

        # labels = convert_class_num(labels)
        def update_bbox(pos, bbox_idx):
            #print(pos.shape, bbox_idx.shape)
            idx = [0]
            bbox = []
            for i in range(1, len(bbox_idx)):
                if bbox_idx[i] != bbox_idx[i - 1]:
                    pos_bbox = pos[idx, :]
                    max_x = pos_bbox[:, 0].max(0)
                    min_x = pos_bbox[:, 0].min(0)
                    max_y = pos_bbox[:, 1].max(0)
                    min_y = pos_bbox[:, 1].min(0)
                    bbox.append([min_x, min_y, max_x, max_y])
                    idx = [i]
                else:
                    idx.append(i)
            pos_bbox = pos[idx, :]
            max_x = pos_bbox[:, 0].max(0)
            min_x = pos_bbox[:, 0].min(0)
            max_y = pos_bbox[:, 1].max(0)
            min_y = pos_bbox[:, 1].min(0)
            bbox.append([min_x, min_y, max_x, max_y])
            return np.array(bbox)

        if self.data_aug:
            pos, bbox, gt_bbox, bbox_targets = self.random_transfer(pos, bbox, gt_bbox, bbox_targets)
            bbox = update_bbox(pos, bbox_idx)
    
        # feats = np.concatenate((
        #     np.zeros((pos.shape[0], 3)),
        #     pos), 
        #     axis = 1)

        feats = np.concatenate((feats, pos), axis = 1)
        ### for plotly
        # scene_feats = np.concatenate((scene_feats, scene_spatial[:, 0:5]), axis = 1)
        ### for cc
        scene_feats = np.concatenate((scene_feats, scene_spatial), axis = 1)

        # print(feats.shape, scene_feats.shape)
        if pos.shape[0] <= 15:
            print(pos.shape, graph_dict['pos']['spatial'].shape, filepath)
            raise SystemError

        def find_k_nearest_neighbors(points, k):
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
            distances, indices = nbrs.kneighbors(points)
            return distances, indices

        distance, indices = find_k_nearest_neighbors(pos, k=15)
        # print(distance)
        aug_feats = np.concatenate((pos, distance[:,4:],), axis = 1)
        
        e_attr = e_attr[:, 0:4]
        e_attr_super = e_attr_super[:, 0:4]

        # if type_edge.shape[0] ==0:

        points = scene_spatial[:, [0, 2]]
        dist_matrix =  squareform(pdist(points))

        type_edge =  []
        for i in range(len(points)):
            # print(i, scene_spatial[i,:], scene_feats[i,6:10])
            for j in range(i+1, len(points)):
                if dist_matrix[i,j] < 0.3 and scene_feats[i, 6:10].all() == scene_feats[j, 6:10].all():
                    type_edge.append(np.array([i, j]))

        # type_edge = np.concatenate([type_edge], axis = 0)

        # if type_edge.shape[0] ==0:
        #     print("No type edge:", filepath)
        #     raise SystemExit

        # print(type_edge.shape, spatial_edge.shape)
        # raise SystemExit
        # e_attr_element = e_attr_element[:, 0:4]
        

        #e_weight = self.getEdgeWeight(pos, edge)
        #e_weight_super = self.getEdgeWeight(pos, edge_super)

        ### covert text data to feat and add to feats
        # texts_f = []
        # for v in value:
        #     if v== None:
        #         text_f = [0, 0]
        #     else:
        #         v = str(v)
        #         if v.isnumeric():
        #             type_ = 1
        #         else:
        #             type_ = 2
        #         length = len(v)
        #         text_f = [type_, length]
        #     texts_f.append(np.array(text_f))
        # texts_f = np.concatenate(texts_f, axis = 0).reshape(-1, 2)
        # print(feats.shape)
        # print(texts_f.shape)
        # feats = np.concatenate([feats, texts_f], axis = 1)
        # print(feats.shape)

        # replace x with aug_feats
        feats = torch.tensor(feats, dtype=torch.float32)
        aug_feats = torch.tensor(aug_feats, dtype=torch.float32)

        # add scene graph
        scene_feats = torch.tensor(scene_feats, dtype=torch.float32)
        spatial_edge = torch.tensor(spatial_edge, dtype=torch.long)
        type_edge = torch.tensor(type_edge, dtype=torch.long)
        # print(type_edge)

        pos = torch.tensor(pos, dtype=torch.float32)
        edge = torch.tensor(edge, dtype=torch.long)
        edge_super = torch.tensor(edge_super, dtype=torch.long)
        # edge_element = torch.tensor(edge_element, dtype=torch.long)
        is_control = torch.tensor(is_control, dtype=torch.bool)
        is_super = torch.tensor(is_super, dtype=torch.bool)
        bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        has_obj = torch.tensor(has_obj, dtype=torch.long)
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        #e_weight = torch.tensor(e_weight, dtype=torch.float32)
        #e_weight_super = torch.tensor(e_weight_super, dtype=torch.float32)
        bbox_idx = torch.tensor(bbox_idx, dtype=torch.long)
        # print(feats.shape, bbox_idx.shape)
        stat_feats = torch.tensor(stat_feats, dtype=torch.float32)
        e_attr = torch.tensor(e_attr, dtype=torch.float32)
        e_attr_super = torch.tensor(e_attr_super, dtype=torch.float32)
        # e_attr_element = torch.tensor(e_attr_element, dtype=torch.float32)


        type_attr = torch.zeros((type_edge.size(0), 4), dtype=torch.float32)

        # raise SystemExit

        data = Data(x = feats, pos = pos)

        # two improvement: relative pos + pixel
        data.aug_feats = aug_feats
        data.value = value
        data.edge = edge
        # data.edge_super = edge_super
        # data.edge_element = edge_element

        # add sence graph
        data.scene_feats = scene_feats
        data.spatial_edge = spatial_edge
        data.type_edge = type_edge
        data.type_attr = type_attr

        data.is_control = is_control
        data.is_super = is_super
        data.bbox = bbox
        data.bbox_targets = bbox_targets
        data.labels = labels
        data.gt_bbox = gt_bbox
        data.gt_labels = gt_labels

        # print(labels.shape, gt_bbox.shape, gt_labels.shape)
        # print(bbox_idx)
        data.filepath = filepath
        data.width = width
        data.height = height
        data.e_attr = e_attr
        data.e_attr_super = e_attr_super
        # data.e_attr_element = e_attr_element

        data.bbox_idx = bbox_idx
        data.stat_feats = stat_feats
        data.has_obj = has_obj
        data.roots = roots
        #data.e_weight = e_weight
        #data.e_weight_super = e_weight_super
        # print(data.x.shape, data.edge.shape, data.e_attr.shape)
        if self.opt.arch == "graphormer":
            data = self.preprocess_item(data)

        return data

    def preprocess_item(self, item):
        #print(item)
        width, height, edge_attr, edge_index, x = item.width, item.height, item.e_attr, item.edge, item.x
        bbox_idx = item.bbox_idx
        edge_index = edge_index.T
        
        N = x.size(0)
        
        #x = convert_to_single_emb(x)

        # node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)]) #, dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            edge_attr #convert_to_single_emb(edge_attr) + 1
        )


        # print(edge_index.shape, edge_index)
        # print(x.shape, edge_attr.shape, edge_index.shape)
        # shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        # max_dist = np.amax(shortest_path_result)
        # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        # spatial_pos = torch.from_numpy((shortest_path_result)).long()
        # attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

        # combine
        item.x = x
        ### set attn_bias as None
        item.attn_bias = None
        item.attn_edge_type = attn_edge_type.long()
        # item.spatial_pos = spatial_pos
        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree  # for undirected graph
        # item.edge_input = torch.from_numpy(edge_input)#.long()
        item.width = width
        item.height = height
        item.bbox_idx = bbox_idx
        # item.adj = adj

        return item
    
    def mixup(self, cc, pos, edge, edge_super, e_attr, e_attr_super, is_super):
        cc_idx = [0 for i in range(len(pos))] 
        cc_edge = [[] for i in range(len(cc))]
        cc_edge_super = [[] for i in range(len(cc))]
        cc_e_attr = [[] for i in range(len(cc))]
        
        for cluster_i, cluster in enumerate(cc):
            for idx in cluster:
                cc_idx[idx] = cluster_i
        for e, a in zip(edge, e_attr):
            cc_edge[cc_idx[e[0]]].append(e)
            cc_e_attr[cc_idx[e[0]]].append(a)
        
        for e in edge_super:
            cc_edge_super[cc_idx[e[0]]].append(e)

        grouped_idx = [[] for i in range(len(cc))]
        offset = pos.shape[0]

        new_cc = []
        new_pos = []
        new_edge = []
        new_edge_super = []
        new_e_attr = []
        new_e_attr_super = []
        new_is_super = []

        def normalize_pos(pos):
            max_x = pos[:, 0].max(0)
            min_x = pos[:, 0].min(0)
            max_y = pos[:, 1].max(0)
            min_y = pos[:, 1].min(0)
            
            if max_x - min_x >  max_y - min_y:
                pos = (pos - [min_x, min_y]) / [max_x - min_x, max_x - min_x]
            else:
                pos = (pos - [min_x, min_y]) / [max_y - min_y, max_y - min_y]
            return pos
        
        def update_edge_idx(edge, old_idx, new_idx):
            o2n = {}
            for i, j in zip(old_idx, new_idx):
                o2n[i] = j
            new_edge = []
            for e in edge:
                new_edge.append([o2n[e[0]], o2n[e[1]]])
            #print(o2n)
            return np.array(new_edge)

        for cluster_i in range(len(cc)):
            cluster_j = random.choice(range(len(cc)))
            cluster = cc[cluster_i]
            cluster_shuffled = cc[cluster_j]
            
            pos_bb0 = pos[cluster]
            pos_bb1 = pos[cluster_shuffled]

            edge_bb0 = np.stack(cc_edge[cluster_i])
            edge_bb1 = np.stack(cc_edge[cluster_j])
            edge_super_bb0 = np.stack(cc_edge_super[cluster_i])
            edge_super_bb1 = np.stack(cc_edge_super[cluster_j])

            e_attr_bb0 = np.stack(cc_e_attr[cluster_i])
            e_attr_bb1 = np.stack(cc_e_attr[cluster_j])
            
            pos_bb0 = normalize_pos(pos_bb0)
            pos_bb1 = normalize_pos(pos_bb1)

            right = random.choice([True, False])
            if right:
                translate_x = 1 + np.random.random() * 0.1
                translate_y = np.random.random()
                pos_bb1[:, 0] += translate_x
                pos_bb1[:, 1] += translate_y
            else:
                translate_x = np.random.random()
                translate_y = 1 +  0.1 * np.random.random()
                pos_bb1[:, 0] += translate_x
                pos_bb1[:, 1] += translate_y

            pos_merged = np.concatenate([pos_bb0, pos_bb1], axis = 0)
            is_super_merged = np.concatenate([is_super[cluster], is_super[cluster_shuffled]], axis = 0)

            idx_bb0 = offset + np.arange(len(cluster))
            idx_bb1 = offset + len(cluster) + np.arange(len(cluster_shuffled))
            #print(idx_bb0, idx_bb1)
            idx_merged = list(idx_bb0) + list(idx_bb1)
            edge_super_merged = []
            for i in idx_bb0:
                for j in idx_bb1:
                    edge_super_merged.append([i, j])
            
            edge_bb0 = update_edge_idx(edge_bb0, cluster, idx_bb0)
            edge_bb1 = update_edge_idx(edge_bb1, cluster_shuffled, idx_bb1)
            edge_super_bb0 = update_edge_idx(edge_super_bb0, cluster, idx_bb0)
            edge_super_bb1 = update_edge_idx(edge_super_bb1, cluster_shuffled, idx_bb1)

            new_pos.append(pos_merged)
            new_cc.append(idx_merged)
            new_edge.append(np.concatenate([edge_bb0, edge_bb1], axis = 0))
            new_edge_super.append(np.concatenate([edge_super_bb0, edge_super_bb1, edge_super_merged], axis = 0))
            new_e_attr.append(np.concatenate([e_attr_bb0, e_attr_bb1], axis = 0))
            new_e_attr_super.append(np.zeros((edge_super_bb0.shape[0] + edge_super_bb1.shape[0] + len(edge_super_merged), 6)))
            new_is_super.append(is_super_merged)


            offset += (len(cluster) + len(cluster_shuffled))
        
        cc = cc + new_cc
        pos = np.concatenate([pos] + new_pos, axis = 0)
        is_super = np.concatenate([is_super] + new_is_super, axis = 0)
        edge = np.concatenate([edge] + new_edge, axis = 0)
        edge_super = np.concatenate([edge_super] + new_edge_super, axis = 0)
        e_attr = np.concatenate([e_attr] + new_e_attr, axis = 0)
        e_attr_super = np.concatenate([e_attr_super] + new_e_attr_super, axis = 0)

        return cc, pos, edge, edge_super, e_attr, e_attr_super, is_super

        
if __name__ == '__main__':
    svg_list = open('/home/xinyangjiang/Datasets/SESYD/FloorPlans/train_list.txt').readlines()
    svg_list = ['/home/xinyangjiang/Datasets/SESYD/FloorPlans/' + line.strip() for line in svg_list]
    builder = SVGGraphBuilderBezier()
    for line in svg_list:
        print(line)
        #line = '/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-01/file_56.svg'
        p = SVGParser(line)
        builder.buildGraph(p.get_all_shape())
