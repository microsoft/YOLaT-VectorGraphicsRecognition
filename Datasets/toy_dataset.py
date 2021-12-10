# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import os
import numpy as np
import copy

from xml.dom.minidom import parse, Node, parseString

from torch_geometric.data import Data
from Datasets.svg_parser import SVGParser, SVGGraphBuilderBezier
from sklearn.metrics.pairwise import euclidean_distances

import random
from Datasets.bezier_parser import BezierParser
#from a2c import a2c

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, root, pre_transform, partition = 'train'):
        super(ToyDataset, self).__init__() 
        
        self.class_dict = {
            'circle': 0, 
            'triangle': 1, 
            'rectangle': 2, 
            #'diamond':1
            #'line': 3
        }

        self.graph_builder = SVGGraphBuilderBezier()

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        key = random.choice(list(self.class_dict.keys()))
        #key = 'line'

        shape_list = []
        shape = {}
        shape['width'] = 1.0
        shape['height'] = 1.0
        shape['stroke'] = 'black'
        shape['stroke-width'] = 3
        
        def _build_shape_line(x1, y1, x2, y2):
            shape_t = copy.copy(shape)
            shape_t['shape_name'] = 'line'
                            
            shape_t['x1'] = x1
            shape_t['y1'] = y1
            shape_t['x2'] = x2
            shape_t['y2'] = y2
            shape_list.append(shape_t)

        if key == 'circle':
            shape['shape_name'] = 'circle'
            shape['cx'] = 0.5
            shape['cy'] = 0.5
            shape['r'] = np.random.uniform(0, 0.5)

            shape_list.append(shape)
        elif key == 'rectangle':
            width = np.random.uniform(0, 1)
            height = np.random.uniform(0, 1)

            _build_shape_line(0, 0, width, 0)
            _build_shape_line(width, 0, width, height)
            _build_shape_line(width, height, 0, height)
            _build_shape_line(0, height, 0, 0)

        elif key == 'triangle':
            x1 = np.random.uniform(0, 1)
            y1 = np.random.uniform(0, 1)
            x2 = np.random.uniform(0, 1)
            y2 = np.random.uniform(0, 1)
            x3 = np.random.uniform(0, 1)
            y3 = np.random.uniform(0, 1)

            _build_shape_line(x1, y1, x2, y2)
            _build_shape_line(x2, y2, x3, y3)
            _build_shape_line(x3, y3, x1, y1)

        elif key == 'line':
            x1 = np.random.uniform(0, 1)
            y1 = np.random.uniform(0, 1)
            x2 = np.random.uniform(0, 1)
            y2 = np.random.uniform(0, 1)

            _build_shape_line(x1, y1, x2, y2)
        
        elif key == 'diamond':
            r = np.random.uniform(0, 0.5)
            x1 = 0.5 - r
            y1 = 0.5
            x2 = 0.5
            y2 = 0.5 - r
            x3 = 0.5 + r 
            y3 = 0.5
            x4 = 0.5
            y4 = 0.5 + r
            _build_shape_line(x1, y1, x2, y2)
            _build_shape_line(x2, y2, x3, y3)
            _build_shape_line(x3, y3, x4, y4)
            _build_shape_line(x4, y4, x1, y1)
            
        #print(shape)
        graph_dict = self.graph_builder.buildGraph(shape_list)
        #print(graph_dict)

        feats = np.concatenate((
            graph_dict['attr']['color'], 
            graph_dict['attr']['stroke_width'], 
            graph_dict['pos']['spatial']), 
            axis = 1)
        feats = graph_dict['pos']['spatial']

        pos = graph_dict['pos']['spatial']
        is_control = graph_dict['attr']['is_control']

        edge = graph_dict['edge']['shape']
        edge_control = graph_dict['edge']['control']
        edge_pos = self.graph_builder.buildPosEdge(pos)

        feats = torch.tensor(feats, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge = torch.tensor(edge, dtype=torch.long)
        edge_pos = torch.tensor(edge_pos, dtype=torch.long)
        edge_control = torch.tensor(edge_control, dtype=torch.long)
        is_control = torch.tensor(is_control, dtype=torch.bool)

        new_pos = []
        new_edge = []

        old2new = {}
        count = 0
        for idx, ic in enumerate(is_control):
            if not ic:
                old2new[idx] = count
                count += 1
        #print(old2new)
        for e in edge:
            if is_control[e[0]] or is_control[e[1]]:
                continue
            new_edge.append((old2new[int(e[0])], old2new[int(e[1])]))
        #print(is_control.size(), pos.size())

        
        feats = feats[~is_control.squeeze()]
        '''
        new_feats = []
        adj = {}
        for e in new_edge:
            if e[0] not in adj:
                adj[e[0]] = []
            if e[1] not in adj:
                adj[e[1]] = []
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        #print(adj)

        #print(feats, 'foo')
        for idx in sorted(list(adj.keys())):
        #    print(idx, adj[idx])
            f = torch.mean(feats[idx] - feats[adj[idx]], dim = 0)
        #    print(f, f.size())
            new_feats.append(f.unsqueeze(0))
        
        feats = torch.cat(new_feats, dim = 0)
        '''

        pos = pos[~is_control.squeeze()]
        is_control = is_control[~is_control.squeeze()]
        edge = torch.tensor(new_edge, dtype=torch.long)

        
        
        labels = self.class_dict[key] * torch.ones((pos.size(0)), dtype=torch.long)
        bbox = torch.zeros((pos.size(0), 4), dtype=torch.float32)
        gt_labels =self.class_dict[key] * torch.ones((pos.size(0)), dtype=torch.long)
        gt_bbox = torch.zeros((pos.size(0), 4), dtype=torch.float32)
        labels = labels[~is_control.squeeze()]
        bbox = bbox[~is_control.squeeze()]
        #print(labels.size(), bbox.size())
        #raise SystemExit

        #print(labels, bbox, pos)
        
        #print('bbox', bbox.size())
        #print('labels', labels.size())
        #raise SystemExit

        data = Data(x = feats, pos = pos)
        data.edge = edge
        data.edge_control = edge_control
        data.edge_pos = edge_pos
        data.is_control = is_control
        data.bbox = bbox
        data.labels = labels
        data.gt_bbox = gt_bbox
        data.gt_labels = gt_labels
        #data.filepath = filepath
        data.width = 1
        data.height = 1
        
        return data
        #elif key == 'triangle':

        #elif key == 'rectangle':
            
        #elif key == 'line':



