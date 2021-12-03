import torch
import os
import numpy as np
from xml.dom.minidom import parse, Node, parseString

from torch_geometric.data import Data
from Datasets.svg_parser import SVGParser, SVGGraphBuilderBezier
from sklearn.metrics.pairwise import euclidean_distances

import networkx as nx
#from a2c import a2c

class SESYDFloorPlan(torch.utils.data.Dataset):
    def __init__(self, root, opt, partition = 'train', data_aug = False):
        super(SESYDFloorPlan, self).__init__() 
        
        svg_list = open(os.path.join(root, partition + '_list.txt')).readlines()
        svg_list = [os.path.join(root, line.strip()) for line in svg_list]
        self.graph_builder = SVGGraphBuilderBezier()
        #print(svg_list)

        self.pos_edge_th = opt.pos_edge_th
        self.data_aug = data_aug

        self.svg_list = svg_list
        
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
            'window2':15
        }
        
        '''
        self.class_dict = {
            'armchair':0, 
            'bed':1, 
            'door1':2, 
            'door2':2, 
            'sink1':3, 
            'sink2':3, 
            'sink3':3, 
            'sink4':3, 
            'sofa1':4, 
            'sofa2':4, 
            'table1':5, 
            'table2':5, 
            'table3':5, 
            'tub':6, 
            'window1':7, 
            'window2':7
        }
        '''
        #self.anchors = self.get_anchor()
        '''
        self.n_objects = 0
        for idx in range(len(self.svg_list)):
            filepath = self.svg_list[idx]
            print(filepath)
            p = SVGParser(filepath)
            width, height = p.get_image_size()
            #graph_dict = self.graph_builder.buildGraph(p.get_all_shape())

            gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
            self.n_objects += gt_bbox.shape[0]
        print(self.n_objects)
        '''
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
        dom = parse(path.replace('.svg', '.xml'))
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

    def gen_y(self, graph_dict, bbox, labels, width, height):
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
        #pos = np.matmul(pos, scale_m[0:2])
        #print(pos.shape)
        center = np.array((0.5, 0.5))[None, :]
        pos -= center
        pos = np.matmul(pos, rot_m[0:2])
        pos += center
        #pos += np.array(translate)[None, :]
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

    def random_transfer(self, pos, bbox, gt_bbox):
        scale = np.random.random() * 0.1 + 0.9
        angle = np.random.random() * np.pi * 2
        translate = [0, 0]
        translate[0] = np.random.random() * 0.2 - 0.1
        translate[1] = np.random.random() * 0.2 - 0.1

        pos = self.__transform__(pos, scale, angle, translate)
        bbox = self.__transform_bbox__(bbox, scale, angle, translate)
        gt_bbox = self.__transform_bbox__(gt_bbox, scale, angle, translate)
        
        return pos, bbox, gt_bbox

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filepath = self.svg_list[idx]
        #filepath = '/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-06/file_97.svg'
        #print(filepath)
        
        p = SVGParser(filepath)
        width, height = p.get_image_size()
        graph_dict = self.graph_builder.buildGraph(p.get_all_shape())       

        gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
        bbox, labels, gt_object = self.gen_y(graph_dict, gt_bbox, gt_labels, width, height)
        
        if self.data_aug:
            graph_dict['pos']['spatial'], bbox, gt_bbox = self.random_transfer(graph_dict['pos']['spatial'], bbox, gt_bbox)

        feats = np.concatenate((
            graph_dict['attr']['color'], 
            #graph_dict['attr']['stroke_width'], 
            graph_dict['pos']['spatial']), 
            axis = 1)
        #feats = graph_dict['pos']['spatial']
        pos = graph_dict['pos']['spatial']
        is_control = graph_dict['attr']['is_control']

        edge = graph_dict['edge']['shape']
        edge_control = graph_dict['edge']['control']
        edge_pos, e_weight_pos = self.graph_builder.buildPosEdge(pos, is_control, th = self.pos_edge_th)
        e_attr = graph_dict['edge_attr']['shape']

        if False:
            top = 2000
            left = 80
            bottom = 2700
            right = 620
            A = np.zeros((pos.shape[0], pos.shape[0]))
            for e in edge:
                print(e)
                p0 = pos[e[0]]
                p1 = pos[e[1]]
                #print(p0, p1)
                p0[0] *= width
                p0[1] *= height
                p1[0] *= width
                p1[1] *= height
                print(p0, p1)
                #raise SystemExit
                #if p0[0] > left and p0[0] < right and p1[1] > top and p1[1] < bottom and p0[0] > left and p0[0] < right and p1[1] > top and p1[1] < bottom:
                #    print('foo')
                #    A[e[0], e[1]] = 1
            G = nx.from_numpy_array(A)
            #print(G.edges)
            raise SystemExit
        
        feats = torch.tensor(feats, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge = torch.tensor(edge, dtype=torch.long)
        edge_pos = torch.tensor(edge_pos, dtype=torch.long)
        edge_control = torch.tensor(edge_control, dtype=torch.long)
        is_control = torch.tensor(is_control, dtype=torch.bool)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        gt_object = torch.tensor(gt_object, dtype=torch.long)

        e_weight = torch.ones(edge.size(0))
        e_weight_control = torch.ones(edge_control.size(0))
        e_weight_pos = torch.tensor(e_weight_pos, dtype=torch.float32)

        e_attr = torch.tensor(e_attr, dtype=torch.float32)
        e_attr_pos = torch.zeros((edge_pos.size(0)), 4, dtype=torch.float32)

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
        data.gt_object = gt_object
        data.filepath = filepath
        data.width = width
        data.height = height
        data.e_weight = e_weight
        data.e_weight_control = e_weight_control
        data.e_weight_pos = e_weight_pos
        data.e_attr = e_attr
        data.e_attr_pos = e_attr_pos
        
        return data

        
if __name__ == '__main__':
    svg_list = open('/home/xinyangjiang/Datasets/SESYD/FloorPlans/train_list.txt').readlines()
    svg_list = ['/home/xinyangjiang/Datasets/SESYD/FloorPlans/' + line.strip() for line in svg_list]
    builder = SVGGraphBuilderBezier()
    for line in svg_list:
        print(line)
        #line = '/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-01/file_56.svg'
        p = SVGParser(line)
        builder.buildGraph(p.get_all_shape())

    #train_dataset = SESYDFloorPlan(opt.data_dir, pre_transform=T.NormalizeScale())
    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    #for batch in train_loader:
    #    pass

    #paths, attributes, svg_attributes = svg2paths2('/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-05/file_47.svg')
    #print(paths, attributes, svg_attributes)
