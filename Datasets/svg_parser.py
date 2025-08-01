# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
from xml.dom.minidom import parse, Node, parseString
from sklearn.metrics.pairwise import euclidean_distances
import copy
from svgpathtools import parse_path, wsvg
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc
from Datasets.bezier_parser import BezierParser
import math
import re
from rich import print


class SVGGraphBuilderBezier3:
    def __init__(self):
        self.bezier_parser = BezierParser()
        self.colors = {
            'black': [0.0, 0.0, 0.0], 
            'red': [1.0, 0.0, 0.0], 
            'green': [0.0, 1.0, 0.0], 
            'blue': [0.0, 0.0, 1.0],
            'white': [1.0, 1.0, 1.0],
            'gray': [0.5, 0.5, 0.5]
        }
        self.types = {
            'path': [1, 0, 0, 0],
            'line': [0, 1, 0, 0],
            'text': [0, 0, 1, 0],
            'rect': [0, 0, 0, 1],
            'circle': [1, 1, 1, 1]
        }
    
    def _build_node_dict(self, node, image_size):
        ret = {}
        img_width, img_height = image_size

        #position
        ret['pos'] = [0, 0]
        ret['pos'][0] = node['pos'][0] / img_width
        ret['pos'][1] = node['pos'][1] / img_height

        #control point
        ret['control_point_start'] = [0, 0]
        ret['control_point_end'] = [0, 0]
            
        #color RGB
        if node['color'] in self.colors:
            ret['color'] = self.colors[node['color']]
        else:
            print('unsuported stroke color!')
            raise SystemExit

        #stroke-width
        ret['stroke-width'] = (node['stroke-width'] - 3) / 3.0
        #print(ret)
        return ret

    def bezierPath2Graph(self, path, attrs, hw):
        edges = []
        # # add edges for one 
        # edges_
        edge_attrs = []
        edges_control = []
        poss = []
        colors = []
        types = []
        values = []
        element_ids = []
        stroke_widths = []
        is_control = []
        

        width = float(hw['width'])
        height = float(hw['height'])

        def hex2rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            return list(int(value[i:i + lv // 3], 16)/255 for i in range(0, lv, lv // 3))

        def _buildNode(point, attr):
            pos = [point.real / width, point.imag / height]

            # fill and stroke 6 dim as color
            colors = []
            for i in range(2):
                if attr['color'][i] in self.colors:
                    color = self.colors[attr['color'][i]]
                elif 'url' in attr['color'][i]:
                    color = [0.5, 0.5, 0.5]
                elif '#' in attr['color'][i]:
                    color = hex2rgb(attr['color'][i])
                elif "rgb" in attr['color'][i]:
                    color = re.findall(r"\d+\.?\d*", attr['color'][i])
                    color = [float(v)/255 for v in color]
                elif attr['color'][i] =="transparent":
                    color = [1, 1, 1]
                elif attr['color'][i] =="currentColor":
                    color = [0, 0, 0]
                else:
                    print(attr['color'][i])
                    print('unsuported stroke color!')
                    raise SystemExit
                colors.extend(color)

            type_ = self.types[attr['type']]

            try:
                value = attr['value']
            except:
                value = None

            stroke_width = (float(attr['stroke-width']) - 3) / 3.0
            element_id = attr['idx']
            #print(pos, color, stroke_width)
            return pos, colors, stroke_width, type_, value, element_id       

        idx = 0
        for element, attr in zip(path, attrs):
            pos_start, color, stroke_width, type_ , value, element_id = _buildNode(element.start, attr)
            poss.append(pos_start)
            colors.append(color)
            types.append(type_)
            values.append(value)
            element_ids.append(element_id)
            stroke_widths.append(stroke_width)
            is_control.append(0)

            pos_c0, color, stroke_width, type_ , value, element_id = _buildNode(element.control1, attr)
            idx_control1 = idx + 1
            poss.append(pos_c0)
            colors.append(color)
            types.append(type_)
            values.append(value)
            element_ids.append(element_id)
            stroke_widths.append(stroke_width)
            is_control.append(1)

            pos_c1, color, stroke_width, type_ , value, element_id = _buildNode(element.control2, attr)
            idx_control2 = idx + 2
            poss.append(pos_c1)
            colors.append(color)
            types.append(type_)
            values.append(value)
            element_ids.append(element_id)
            stroke_widths.append(stroke_width)
            is_control.append(1)

            pos_end, color, stroke_width, type_, value, element_id = _buildNode(element.end, attr)
            idx_end = idx + 3
            poss.append(pos_end)
            colors.append(color)
            types.append(type_)
            values.append(value)
            element_ids.append(element_id)
            stroke_widths.append(stroke_width)
            is_control.append(0)

            '''
            cross_prod = pos_start[0] * pos_end[1] - pos_start[1] * pos_end[0]
            if cross_prod < 0:
                angle = 2 * np.pi - angle
            '''

            euc_d2 = (pos_start[0] - pos_end[0]) * (pos_start[0] - pos_end[0]) + (pos_start[1] - pos_end[1]) * (pos_start[1] - pos_end[1])
            angle = (pos_start[0] - pos_end[0]) / (np.sqrt(euc_d2) + 1e-7)
            w = 1 / np.exp(euc_d2)
            #w = (w - 0.8) / 0.2
            #print(angle, w)
            if math.isnan(angle):
                print(angle, pos_start, pos_end, dot_prod, cos_theta)
                raise SystemExit


            edges.append([idx, idx_end])
            # print(element, [idx, idx_end])
            edges_control.append([idx, idx_control1])
            edges_control.append([idx, idx_control2])
            edges_control.append([idx_end, idx_control2])
            edges_control.append([idx_end, idx_control1])
            edges_control.append([idx_control1, idx_control2])


            edge_attrs.append([pos_c0[0] - pos_start[0], 
                pos_c0[1] - pos_start[1], 
                pos_c1[0] - pos_end[0], 
                pos_c1[1] - pos_end[1], 
                angle, 
                euc_d2
                ])

            idx += 4
        
        #print (poss, colors, stroke_widths, edges_control, edges)
        #)
        
        return {'pos':{'spatial': poss}, 
            'attr':{'color': colors, 'stroke_width': stroke_widths, 'type': types, 'is_control':is_control, 'value': values, 'idx': element_ids}, 
            'edge': {'shape':edges, 'control': edges_control}, 
            'edge_attr':{'shape': edge_attrs}
        }

    def mergeNode(self, graph_dict):
        pos = graph_dict['pos']['spatial']
        sim_pos = euclidean_distances(pos, pos)
        #np.set_printoptions(threshold=np.inf)
        sim_pos = (sim_pos < 1e-3) # change 1e-3 to 1e-8

        is_control = graph_dict['attr']['is_control']
        
        sim_attr = np.ones((pos.shape[0], pos.shape[0])).astype(bool)

        # del graph_dict['attr']['type']
        # del graph_dict['attr']['idx']

        # print(graph_dict['attr'])
        sim_attr = np.ones((pos.shape[0], pos.shape[0])).astype(bool)
        for key in graph_dict['attr']:
            if key != "value":
                s = euclidean_distances(graph_dict['attr'][key], graph_dict['attr'][key])
                s = (s < 1e-6)
                #print(s)
                sim_attr = sim_attr & s      

        #sim_att = (sim_att > 0)

        # similar node: similar pos & attr & is end point
        sim = sim_pos * sim_attr * (is_control == 0)
        # print('sim', sim)
        # print('sim_pos', sim_pos)
        # print('sim_attr', sim_attr)
        # print('is_control', is_control)
        # print('pos', graph_dict['pos']['spatial'])
        #sim2 = sim_pos * (is_control < 0) * (is_control.T < 0)
        #print(np.sum(sim ^ sim2))
        #print(sim, sim2)

        #create sim node set
        n_node = pos.shape[0]        
        visited = [False] * n_node
        clusters = []
        for start_node in range(0, n_node):
            if visited[start_node]: continue
            
            cluster = [start_node]
            visited[start_node] = True
            queue = [start_node]

            while len(queue) != 0:
                node_idx = queue.pop(0)
                neighbors = sim[node_idx]
                for i in range(0, n_node):
                    if neighbors[i] and not visited[i]:
                        cluster.append(i)
                        visited[i] = True
                        queue.append(i)
       
            clusters.append(cluster)

        '''
        n_total = 0
        selected = [0] * n_node
        for cluster in clusters:
            print(cluster)
            n_total += len(cluster)
            for n in cluster:
                selected[n] += 1
        print([(i, n) for i, n in enumerate(selected) if n != 1])
        print(n_total, n_node)
        '''
        
        merging_map = list(range(0, n_node))
        for new_idx, cluster in enumerate(clusters):
            for n in cluster:
                merging_map[n] = new_idx
        n_cluster = len(clusters)

        #print(merging_map)
        merged_graph_dict = {}
        oldedge2newedge = {}
        for key in graph_dict:
            if key == 'edge_attr': continue
            merged_graph_dict[key] = {}
            if key == 'edge':
                for k in graph_dict[key]:
                    if k != 'shape':
                        merged_edge = set()
                        for e in graph_dict[key][k]:
                            #print(e)
                            if merging_map[e[0]] != merging_map[e[1]]:
                                merged_e = tuple(sorted([merging_map[e[0]], merging_map[e[1]]]))
                            #print(merged_e)
                            merged_edge.add(merged_e)
                        merged_graph_dict[key][k] = np.array(list(merged_edge))
                    elif k == 'shape':
                        merged_edge = set()
                        merged_edge_attr_dict = {}
                        for e, e_attr in zip(graph_dict[key][k], graph_dict['edge_attr']['shape']):
                            if merging_map[e[0]] == merging_map[e[1]]: continue
                            merged_e = tuple(sorted([merging_map[e[0]], merging_map[e[1]]]))
                            if merged_e not in merged_edge_attr_dict:
                                merged_edge_attr_dict[merged_e] = []
                            merged_edge.add(merged_e)
                            merged_edge_attr_dict[merged_e].append(e_attr)
                        merged_graph_dict[key][k] = np.array(list(merged_edge))

                        merged_edge_attr = []
                        for e in merged_graph_dict[key][k]:
                            ea = np.array(merged_edge_attr_dict[tuple(e)])
                            #if ea.shape[0] > 1:
                            #    print(ea.shape, ea)
                            ea = np.mean(ea, axis = 0)
                            #print(ea)
                            #raise SystemExit
                            merged_edge_attr.append(ea)
                        
                        merged_graph_dict['edge_attr'] = {}
                        merged_graph_dict['edge_attr']['shape'] = np.array(merged_edge_attr)
                        
                        #print(merged_graph_dict['edge_attr']['shape'].shape)
                        #print(merged_graph_dict['edge']['shape'].shape)
                        #raise SystemExit
            else:
                for k in graph_dict[key]:
                    if k == "value": 
                        mat = graph_dict[key][k]
                        merged_mat = []
                        for i in range(0, n_cluster):
                            merged_mat.append(mat[clusters[i][0]])
                        merged_graph_dict[key][k] = merged_mat 
                    else:
                        mat = graph_dict[key][k]
                        merged_mat = np.zeros((n_cluster, mat.shape[1]))
                        for i in range(0, n_cluster):
                            merged_mat[i] = np.mean(mat[clusters[i]], axis = 0)
                        merged_graph_dict[key][k] = merged_mat 

        return merged_graph_dict



        '''
        print(sim_pos * sim_att)
        print(sim)

        print(pos[0], 'query')
        for i, s in enumerate(sim[0]):
            if s:
                print(graph_dict['pos']['spatial'][i], graph_dict['attr']['is_control'][i])
        '''

    def buildPosEdge(self, pos, is_control, th = 5e-3):
        distance = euclidean_distances(pos, pos)
        s = (distance < th)
        
        ret = []
        weight = []

        for idx in range(0, s.shape[0]):
            w_sum = 0
            w_list = []
            for i, ss in enumerate(s[idx]):
                if ss and (not is_control[idx]) and (not is_control[i]) and i != idx:
                    ret.append([idx, i])
                    w = 1 - distance[idx, i]
                    w_list.append(w)
                    w_sum += w
            w_list = [w / w_sum for w in w_list]
            weight += w_list

        return np.array(ret), np.array(weight)

    def buildGraph(self, shape_list):
        graph_dict = {}
        idx_offset = 0
        for shape in shape_list:
            bezier_path = self.bezier_parser.shape2BezierPath(shape)
            node_dict = self.bezierPath2Graph(bezier_path, shape)

            #if len(node_dict['pos']['spatial']) <= np.max(node_dict['edge']['shape']) or len(node_dict['pos']['spatial']) <= np.max(node_dict['edge']['control']):
            #    print(shape['shape_name'], node_dict)

            for key in node_dict:
                if key not in graph_dict:
                    graph_dict[key] = {}
                for k in node_dict[key]:
                    if k not in graph_dict[key]:
                        graph_dict[key][k] = []
                    if key == 'edge':
                        e = np.array(node_dict[key][k]) + idx_offset
                        graph_dict[key][k].append(e)
                    else:
                        graph_dict[key][k].append(np.array(node_dict[key][k]))

            idx_offset += len(node_dict['pos']['spatial'])
        
        for key in graph_dict:
            for k in graph_dict[key]:
                graph_dict[key][k] = np.concatenate(graph_dict[key][k], axis = 0)
                if len(graph_dict[key][k].shape) == 1:
                    graph_dict[key][k] = graph_dict[key][k][:, None]
                #print(key, k,  graph_dict[key][k].shape)

        graph_dict = self.mergeNode(graph_dict)
        
        return graph_dict


class SVGGraphBuilderBezier2:
    def __init__(self):
        self.bezier_parser = BezierParser()
        self.colors = {
            'black': [0.0, 0.0, 0.0], 
            'red': [1.0, 0.0, 0.0], 
            'green': [0.0, 1.0, 0.0], 
            'blue': [0.0, 0.0, 1.0],
            'gray': [0.5, 0.5, 0.5]
        }
    
    def _build_node_dict(self, node, image_size):
        ret = {}
        img_width, img_height = image_size

        #position
        ret['pos'] = [0, 0]
        ret['pos'][0] = node['pos'][0] / img_width
        ret['pos'][1] = node['pos'][1] / img_height

        #control point
        ret['control_point_start'] = [0, 0]
        ret['control_point_end'] = [0, 0]
            
        #color RGB
        if node['color'] in self.colors:
            ret['color'] = self.colors[node['color']]
        else:
            print('unsuported stroke color!')
            raise SystemExit

        #stroke-width
        ret['stroke-width'] = (node['stroke-width'] - 3) / 3.0
        #print(ret)
        return ret

    def bezierPath2Graph(self, path, attrs):
        edges = []
        edge_attrs = []
        edges_control = []
        poss = []
        colors = []
        stroke_widths = []
        is_control = []

        width = float(attrs['width'])
        height = float(attrs['height'])
        def _buildNode(point):
            pos = [point.real / width, point.imag / height]
            if attrs['stroke'] in self.colors:
                color = self.colors[attrs['stroke']]
            else:
                print(attrs['stroke'])
                print('unsuported stroke color!')
                raise SystemExit

            stroke_width = (float(attrs['stroke-width']) - 3) / 3.0
            #print(pos, color, stroke_width)
            return pos, color, stroke_width       

        idx = 0
        for element in path:
            pos_start, color, stroke_width = _buildNode(element.start)
            poss.append(pos_start)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(0)

            pos_c0, color, stroke_width = _buildNode(element.control1)
            idx_control1 = idx + 1
            poss.append(pos_c0)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(1)

            pos_c1, color, stroke_width = _buildNode(element.control2)
            idx_control2 = idx + 2
            poss.append(pos_c1)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(1)

            pos_end, color, stroke_width = _buildNode(element.end)
            idx_end = idx + 3
            poss.append(pos_end)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(0)

            
            

            '''
            cross_prod = pos_start[0] * pos_end[1] - pos_start[1] * pos_end[0]
            if cross_prod < 0:
                angle = 2 * np.pi - angle
            '''

            euc_d2 = (pos_start[0] - pos_end[0]) * (pos_start[0] - pos_end[0]) + (pos_start[1] - pos_end[1]) * (pos_start[1] - pos_end[1])
            angle = (pos_start[0] - pos_end[0]) / (np.sqrt(euc_d2) + 1e-7)
            w = 1 / np.exp(euc_d2)
            #w = (w - 0.8) / 0.2
            #print(angle, w)
            if math.isnan(angle):
                print(angle, pos_start, pos_end, dot_prod, cos_theta)
                raise SystemExit

            edges.append([idx, idx_end])
            edges_control.append([idx, idx_control1])
            edges_control.append([idx, idx_control2])
            edges_control.append([idx_end, idx_control2])
            edges_control.append([idx_end, idx_control1])
            edges_control.append([idx_control1, idx_control2])


            edge_attrs.append([pos_c0[0] - pos_start[0], 
                pos_c0[1] - pos_start[1], 
                pos_c1[0] - pos_end[0], 
                pos_c1[1] - pos_end[1], 
                angle, 
                euc_d2
                ])

            idx += 4
        
        #print (poss, colors, stroke_widths, edges_control, edges)
        #)
        
        return {'pos':{'spatial': poss}, 
            'attr':{'color': colors, 'stroke_width': stroke_widths, 'is_control':is_control}, 
            'edge': {'shape':edges, 'control': edges_control}, 
            'edge_attr':{'shape': edge_attrs}
        }

    def mergeNode(self, graph_dict):
        pos = graph_dict['pos']['spatial']
        sim_pos = euclidean_distances(pos, pos)
        #np.set_printoptions(threshold=np.inf)
        sim_pos = (sim_pos < 1e-3)

        is_control = graph_dict['attr']['is_control']
        
        sim_attr = np.ones((pos.shape[0], pos.shape[0])).astype(bool)
        for key in graph_dict['attr']:
            s = euclidean_distances(graph_dict['attr'][key], graph_dict['attr'][key])
            s = (s < 1e-8)
            #print(s)
            sim_attr = sim_attr & s      

        #sim_att = (sim_att > 0)

        # similar node: similar pos & attr & is end point
        sim = sim_pos * sim_attr * (is_control == 0)
        # print('sim', sim)
        # print('sim_pos', sim_pos)
        # print('sim_attr', sim_attr)
        # print('is_control', is_control)
        #print('pos', graph_dict['pos']['spatial'])
        #sim2 = sim_pos * (is_control < 0) * (is_control.T < 0)
        #print(np.sum(sim ^ sim2))
        #print(sim, sim2)

        #create sim node set
        n_node = pos.shape[0]        
        visited = [False] * n_node
        clusters = []
        for start_node in range(0, n_node):
            if visited[start_node]: continue
            
            cluster = [start_node]
            visited[start_node] = True
            queue = [start_node]

            while len(queue) != 0:
                node_idx = queue.pop(0)
                neighbors = sim[node_idx]
                for i in range(0, n_node):
                    if neighbors[i] and not visited[i]:
                        cluster.append(i)
                        visited[i] = True
                        queue.append(i)
       
            clusters.append(cluster)

        '''
        n_total = 0
        selected = [0] * n_node
        for cluster in clusters:
            print(cluster)
            n_total += len(cluster)
            for n in cluster:
                selected[n] += 1
        print([(i, n) for i, n in enumerate(selected) if n != 1])
        print(n_total, n_node)
        '''
        
        merging_map = list(range(0, n_node))
        for new_idx, cluster in enumerate(clusters):
            for n in cluster:
                merging_map[n] = new_idx
        n_cluster = len(clusters)

        #print(merging_map)
        merged_graph_dict = {}
        oldedge2newedge = {}
        for key in graph_dict:
            if key == 'edge_attr': continue
            merged_graph_dict[key] = {}
            if key == 'edge':
                for k in graph_dict[key]:
                    if k != 'shape':
                        merged_edge = set()
                        for e in graph_dict[key][k]:
                            #print(e)
                            if merging_map[e[0]] != merging_map[e[1]]:
                                merged_e = tuple(sorted([merging_map[e[0]], merging_map[e[1]]]))
                            #print(merged_e)
                            merged_edge.add(merged_e)
                        merged_graph_dict[key][k] = np.array(list(merged_edge))
                    elif k == 'shape':
                        merged_edge = set()
                        merged_edge_attr_dict = {}
                        for e, e_attr in zip(graph_dict[key][k], graph_dict['edge_attr']['shape']):
                            if merging_map[e[0]] == merging_map[e[1]]: continue
                            merged_e = tuple(sorted([merging_map[e[0]], merging_map[e[1]]]))
                            if merged_e not in merged_edge_attr_dict:
                                merged_edge_attr_dict[merged_e] = []
                            merged_edge.add(merged_e)
                            merged_edge_attr_dict[merged_e].append(e_attr)
                        merged_graph_dict[key][k] = np.array(list(merged_edge))

                        merged_edge_attr = []
                        for e in merged_graph_dict[key][k]:
                            ea = np.array(merged_edge_attr_dict[tuple(e)])
                            #if ea.shape[0] > 1:
                            #    print(ea.shape, ea)
                            ea = np.mean(ea, axis = 0)
                            #print(ea)
                            #raise SystemExit
                            merged_edge_attr.append(ea)
                        
                        merged_graph_dict['edge_attr'] = {}
                        merged_graph_dict['edge_attr']['shape'] = np.array(merged_edge_attr)
                        
                        #print(merged_graph_dict['edge_attr']['shape'].shape)
                        #print(merged_graph_dict['edge']['shape'].shape)
                        #raise SystemExit
            else:
                for k in graph_dict[key]:
                    mat = graph_dict[key][k]
                    merged_mat = np.zeros((n_cluster, mat.shape[1]))
                    for i in range(0, n_cluster):
                        merged_mat[i] = np.mean(mat[clusters[i]], axis = 0)
                    merged_graph_dict[key][k] = merged_mat                        
        
        return merged_graph_dict

        '''
        print(sim_pos * sim_att)
        print(sim)

        print(pos[0], 'query')
        for i, s in enumerate(sim[0]):
            if s:
                print(graph_dict['pos']['spatial'][i], graph_dict['attr']['is_control'][i])
        '''

    def buildPosEdge(self, pos, is_control, th = 5e-3):
        distance = euclidean_distances(pos, pos)
        s = (distance < th)
        
        ret = []
        weight = []

        for idx in range(0, s.shape[0]):
            w_sum = 0
            w_list = []
            for i, ss in enumerate(s[idx]):
                if ss and (not is_control[idx]) and (not is_control[i]) and i != idx:
                    ret.append([idx, i])
                    w = 1 - distance[idx, i]
                    w_list.append(w)
                    w_sum += w
            w_list = [w / w_sum for w in w_list]
            weight += w_list

        return np.array(ret), np.array(weight)

    def buildGraph(self, shape_list):
        graph_dict = {}
        idx_offset = 0
        for shape in shape_list:
            bezier_path = self.bezier_parser.shape2BezierPath(shape)
            node_dict = self.bezierPath2Graph(bezier_path, shape)

            #if len(node_dict['pos']['spatial']) <= np.max(node_dict['edge']['shape']) or len(node_dict['pos']['spatial']) <= np.max(node_dict['edge']['control']):
            #    print(shape['shape_name'], node_dict)

            for key in node_dict:
                if key not in graph_dict:
                    graph_dict[key] = {}
                for k in node_dict[key]:
                    if k not in graph_dict[key]:
                        graph_dict[key][k] = []
                    if key == 'edge':
                        e = np.array(node_dict[key][k]) + idx_offset
                        graph_dict[key][k].append(e)
                    else:
                        graph_dict[key][k].append(np.array(node_dict[key][k]))

            idx_offset += len(node_dict['pos']['spatial'])
        
        for key in graph_dict:
            for k in graph_dict[key]:
                graph_dict[key][k] = np.concatenate(graph_dict[key][k], axis = 0)
                if len(graph_dict[key][k].shape) == 1:
                    graph_dict[key][k] = graph_dict[key][k][:, None]
                #print(key, k,  graph_dict[key][k].shape)

        graph_dict = self.mergeNode(graph_dict)
        
        return graph_dict

class SVGGraphBuilderShape:
    def __init__(self):
        self.bezier_parser = BezierParser()
        self.colors = {
            'black': [0.0, 0.0, 0.0], 
            'red': [1.0, 0.0, 0.0], 
            'green': [0.0, 1.0, 0.0], 
            'blue': [0.0, 0.0, 1.0]
        }
    
    def buildPosEdge(self, pos, th = 5e-3):
        #d00 = euclidean_distances(pos[:, 0:2], pos[:, 0:2])
        #d01 = euclidean_distances(pos[:, 0:2], pos[:, 2:])
        #d10 = euclidean_distances(pos[:, 2:], pos[:, 0:2])
        #d11 = euclidean_distances(pos[:, 2:], pos[:, 2:])

        #distance = np.concatenate([d00[:, :, None], d01[:, :, None], d10[:, :, None], d11[:, :, None]], axis = 2)
        #distance = distance.min(axis = 2)
        #s = (distance < th)
        distance = euclidean_distances(pos, pos)
        s = (distance < th)

        ret = []
        weight = []

        for idx in range(0, s.shape[0]):
            w_sum = 0
            w_list = []
            for i, ss in enumerate(s[idx]):
                if ss:
                    ret.append([idx, i])
                    w = 1 - distance[idx, i]
                    w_list.append(w)
                    w_sum += w
            w_list = [w / w_sum for w in w_list]
            weight += w_list

        return np.array(ret), np.array(weight)

    def buildGraph(self, shape_list):
        graph_dict = {}
        idx_offset = 0

        feats = []
        poses = []
        for shape in shape_list:
            #print(shape)
            width = float(shape['width'])
            height = float(shape['height'])
            
            if shape['shape_name'] == 'line':
                node = {}
                x1 = float(shape['x1']) / width
                x2 = float(shape['x2']) / width
                y1 = float(shape['y1']) / height
                y2 = float(shape['y2']) / height                
                f = [x1, y1, x2, y2]
                pos = [(x1 + x2) / 2 , (y1 + y2) /2]
                f_array = np.zeros((1, 17))
                f_array[:, 0:4] = f
                node['pos'] = pos
                feats.append(f_array)
                poses.append(pos)

            elif shape['shape_name'] == 'path':
                path = parse_path(shape['d'])
                bezier_path = Path()
                for element in path:
                    if isinstance(element, Arc):
                        arc_path = element
                        #wsvg(shape, filename = 'arc.svg')
                        #print(path)
                        x1 = arc_path.start.real / width
                        y1 = arc_path.start.imag / height
                        x2 = arc_path.end.real / width
                        y2 = arc_path.end.imag / height
                        rx = arc_path.radius.real / width
                        ry = arc_path.radius.imag / height
                        phi = arc_path.rotation
                        if arc_path.large_arc: fa = 1.0
                        else: fa = 0.0
                        if arc_path.sweep: fs = 1.0
                        else: fs = 0.0

                        f = (x1, y1, x2, y2, rx, ry, phi, fa, fs)
                        pos = [(x1 + x2) / 2 , (y1 + y2) /2]
                        f_array[:, 4:13] = f
                        node['pos'] = pos
                        feats.append(f_array)
                        poses.append(pos)

            elif shape['shape_name'] == 'circle':
                #print(shape)
                cx = float(shape['cx']) / width
                cy = float(shape['cy']) / height
                rx = float(shape['r']) / width
                ry = float(shape['r']) / height
                f = (cx, cy, rx, ry)
                pos = (cx, cy)
                f_array[:, 13:] = f
                node['pos'] = pos
                feats.append(f_array)
                poses.append(pos)

            else:
                print(shape)
                raise SystemExit

        #graph_dict = self.mergeNode(graph_dict)
        graph_dict = {}
        graph_dict['f'] = np.concatenate(feats, axis = 0)
        graph_dict['pos'] = np.array(poses)
        #print(graph_dict['f'].shape)
        #print(graph_dict['pos'].shape)

        edge, edge_weight = self.buildPosEdge(graph_dict['pos'])
        #print(edge, edge_weight)
        graph_dict['edge'] = edge
        graph_dict['edge_weight'] = edge_weight

        #raise SystemExit

        return graph_dict

class SVGGraphBuilderBezier:
    def __init__(self):
        self.bezier_parser = BezierParser()
        self.colors = {
            'black': [0.0, 0.0, 0.0], 
            'red': [1.0, 0.0, 0.0], 
            'green': [0.0, 1.0, 0.0], 
            'blue': [0.0, 0.0, 1.0]
        }
    
    def _build_node_dict(self, node, image_size):
        ret = {}
        img_width, img_height = image_size

        #position
        ret['pos'] = [0, 0]
        ret['pos'][0] = node['pos'][0] / img_width
        ret['pos'][1] = node['pos'][1] / img_height

        #control point
        ret['control_point_start'] = [0, 0]
        ret['control_point_end'] = [0, 0]
            
        #color RGB
        if node['color'] in self.colors:
            ret['color'] = self.colors[node['color']]
        else:
            print('unsuported stroke color!')
            raise SystemExit

        #stroke-width
        ret['stroke-width'] = (node['stroke-width'] - 3) / 3.0
        #print(ret)
        return ret

    def bezierPath2Graph(self, path, attrs):
        edges = []
        edge_attrs = []
        edges_control = []
        poss = []
        colors = []
        stroke_widths = []
        is_control = []

        width = float(attrs['width'])
        height = float(attrs['height'])
        def _buildNode(point):
            pos = [point.real / width, point.imag / height]
            
            if attrs['stroke'] in self.colors:
                color = self.colors[attrs['stroke']]
            else:
                print('unsuported stroke color!')
                raise SystemExit

            stroke_width = (float(attrs['stroke-width']) - 3) / 3.0
            #print(pos, color, stroke_width)
            return pos, color, stroke_width       

        idx = 0
        for element in path:
            pos_start, color, stroke_width = _buildNode(element.start)
            poss.append(pos_start)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(0)

            pos_c0, color, stroke_width = _buildNode(element.control1)
            idx_control1 = idx + 1
            poss.append(pos_c0)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(1)

            pos_c1, color, stroke_width = _buildNode(element.control2)
            idx_control2 = idx + 2
            poss.append(pos_c1)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(1)

            pos_end, color, stroke_width = _buildNode(element.end)
            idx_end = idx + 3
            poss.append(pos_end)
            colors.append(color)
            stroke_widths.append(stroke_width)
            is_control.append(0)

            edges.append([idx, idx_end])
            edges_control.append([idx, idx_control1])
            edges_control.append([idx, idx_control2])
            edges_control.append([idx_end, idx_control2])
            edges_control.append([idx_end, idx_control1])
            edges_control.append([idx_control1, idx_control2])

            edge_attrs.append([pos_c0[0] - pos_start[0], 
                pos_c0[1] - pos_start[1], 
                pos_c1[0] - pos_end[0], 
                pos_c1[1] - pos_end[1]
                ])

            idx += 4
        
        #print (poss, colors, stroke_widths, edges_control, edges)
        #)
        
        return {'pos':{'spatial': poss}, 
            'attr':{'color': colors, 'stroke_width': stroke_widths, 'is_control':is_control}, 
            'edge': {'shape':edges, 'control': edges_control}, 
            'edge_attr':{'shape': edge_attrs}
        }

    def mergeNode(self, graph_dict):
        pos = graph_dict['pos']['spatial']
        sim_pos = euclidean_distances(pos, pos)
        #print('euc dist', sim_pos)
        #np.set_printoptions(threshold=np.inf)
        sim_pos = (sim_pos < 1e-3) #1e-7)

        is_control = graph_dict['attr']['is_control']
        
        sim_attr = np.ones((pos.shape[0], pos.shape[0])).astype(bool)
        for key in graph_dict['attr']:
            #print(key)
            s = euclidean_distances(graph_dict['attr'][key], graph_dict['attr'][key])
            s = (s < 1e-8)
            #print(s)
            sim_attr = sim_attr & s      

        #sim_att = (sim_att > 0)

        sim = sim_pos * sim_attr * (is_control == 0)
        #print('sim', sim)
        #print('sim_pos', sim_pos)
        #print('sim_attr', sim_attr)
        #print('is_control', is_control)
        #print('pos', graph_dict['pos']['spatial'])
        #sim2 = sim_pos * (is_control < 0) * (is_control.T < 0)
        #print(np.sum(sim ^ sim2))
        #print(sim, sim2)

        #create sim node set
        n_node = pos.shape[0]        
        visited = [False] * n_node
        clusters = []
        for start_node in range(0, n_node):
            if visited[start_node]: continue
            
            cluster = [start_node]
            visited[start_node] = True
            queue = [start_node]

            while len(queue) != 0:
                node_idx = queue.pop(0)
                neighbors = sim[node_idx]
                for i in range(0, n_node):
                    if neighbors[i] and not visited[i]:
                        cluster.append(i)
                        visited[i] = True
                        queue.append(i)
       
            clusters.append(cluster)

        '''
        n_total = 0
        selected = [0] * n_node
        for cluster in clusters:
            print(cluster)
            n_total += len(cluster)
            for n in cluster:
                selected[n] += 1
        print([(i, n) for i, n in enumerate(selected) if n != 1])
        print(n_total, n_node)
        '''
        
        merging_map = list(range(0, n_node))
        for new_idx, cluster in enumerate(clusters):
            for n in cluster:
                merging_map[n] = new_idx
        n_cluster = len(clusters)

        #print(merging_map)
        merged_graph_dict = {}
        oldedge2newedge = {}
        for key in graph_dict:
            if key == 'edge_attr': continue
            merged_graph_dict[key] = {}
            if key == 'edge':
                for k in graph_dict[key]:
                    if k != 'shape':
                        merged_edge = set()
                        for e in graph_dict[key][k]:
                            #print(e)
                            if merging_map[e[0]] != merging_map[e[1]]:
                                merged_e = tuple(sorted([merging_map[e[0]], merging_map[e[1]]]))
                            #print(merged_e)
                            merged_edge.add(merged_e)
                        merged_graph_dict[key][k] = np.array(list(merged_edge))
                    elif k == 'shape':
                        merged_edge = set()
                        merged_edge_attr_dict = {}
                        for e, e_attr in zip(graph_dict[key][k], graph_dict['edge_attr']['shape']):
                            merged_e = tuple(sorted([merging_map[e[0]], merging_map[e[1]]]))
                            if merged_e not in merged_edge_attr_dict:
                                merged_edge_attr_dict[merged_e] = []
                            merged_edge.add(merged_e)
                            merged_edge_attr_dict[merged_e].append(e_attr)
                        merged_graph_dict[key][k] = np.array(list(merged_edge))

                        merged_edge_attr = []
                        for e in merged_graph_dict[key][k]:
                            ea = np.array(merged_edge_attr_dict[tuple(e)])
                            #if ea.shape[0] > 1:
                            #    print(ea.shape, ea)
                            ea = np.mean(ea, axis = 0)
                            #print(ea)
                            #raise SystemExit
                            merged_edge_attr.append(ea)
                        
                        merged_graph_dict['edge_attr'] = {}
                        merged_graph_dict['edge_attr']['shape'] = np.array(merged_edge_attr)
                        
                        #print(merged_graph_dict['edge_attr']['shape'].shape)
                        #print(merged_graph_dict['edge']['shape'].shape)
                        #raise SystemExit
            else:
                for k in graph_dict[key]:
                    mat = graph_dict[key][k]
                    merged_mat = np.zeros((n_cluster, mat.shape[1]))
                    for i in range(0, n_cluster):
                        merged_mat[i] = np.mean(mat[clusters[i]], axis = 0)
                    merged_graph_dict[key][k] = merged_mat                        
        
        return merged_graph_dict



        '''
        print(sim_pos * sim_att)
        print(sim)

        print(pos[0], 'query')
        for i, s in enumerate(sim[0]):
            if s:
                print(graph_dict['pos']['spatial'][i], graph_dict['attr']['is_control'][i])
        '''

    def buildPosEdge(self, pos, is_control, th = 5e-3):
        distance = euclidean_distances(pos, pos)
        s = (distance < th)
        
        ret = []
        weight = []

        for idx in range(0, s.shape[0]):
            w_sum = 0
            w_list = []
            for i, ss in enumerate(s[idx]):
                if ss and (not is_control[idx]) and (not is_control[i]) and i != idx:
                    ret.append([idx, i])
                    w = 1 - distance[idx, i]
                    w_list.append(w)
                    w_sum += w
            w_list = [w / w_sum for w in w_list]
            weight += w_list

        return np.array(ret), np.array(weight)

    def buildGraph(self, shape_list):
        graph_dict = {}
        idx_offset = 0
        for shape in shape_list:
            bezier_path = self.bezier_parser.shape2BezierPath(shape)
            node_dict = self.bezierPath2Graph(bezier_path, shape)

            #if len(node_dict['pos']['spatial']) <= np.max(node_dict['edge']['shape']) or len(node_dict['pos']['spatial']) <= np.max(node_dict['edge']['control']):
            #    print(shape['shape_name'], node_dict)

            for key in node_dict:
                if key not in graph_dict:
                    graph_dict[key] = {}
                for k in node_dict[key]:
                    if k not in graph_dict[key]:
                        graph_dict[key][k] = []
                    if key == 'edge':
                        e = np.array(node_dict[key][k]) + idx_offset
                        graph_dict[key][k].append(e)
                    else:
                        graph_dict[key][k].append(np.array(node_dict[key][k]))

            idx_offset += len(node_dict['pos']['spatial'])
        
        for key in graph_dict:
            for k in graph_dict[key]:
                graph_dict[key][k] = np.concatenate(graph_dict[key][k], axis = 0)
                if len(graph_dict[key][k].shape) == 1:
                    graph_dict[key][k] = graph_dict[key][k][:, None]
                #print(key, k,  graph_dict[key][k].shape)

        graph_dict = self.mergeNode(graph_dict)
        
        return graph_dict

class SVGParser:
    def __init__(self, filepath):
        self.dom = parse(filepath)
        self.root = self.dom.documentElement
        self.shapes = ['line', 'path', 'circle']
        self.filtered_nodename = ['image', 'g', 'defs']

    def _traverse_tree(self, root, ret_list, parent_attrs):
        parent_attrs = copy.copy(parent_attrs)
        
        if root.attributes is not None:
            attrs = root.attributes.items()
            for att in attrs:
                parent_attrs[att[0]] = att[1]

        for child in root.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                if child.nodeName in self.shapes:
                    attrs = child.attributes.items()
                    #print(attrs, child.nodeType)
                    attr_dict = copy.copy(parent_attrs)
                    for att in attrs:
                        attr_dict[att[0]] = att[1]
                    attr_dict['shape_name'] = child.nodeName
                    ret_list.append(attr_dict)
                elif child.nodeName not in self.filtered_nodename:
                    print('node is not a supported shape', child.tagName)
                    raise SystemExit
            self._traverse_tree(child, ret_list, parent_attrs)

    def get_all_shape(self):
        #obtain all shape
        ret_list = []
        self._traverse_tree(self.root, ret_list, {})
        return ret_list

    def get_image_size(self):
        img_info = self.root.getElementsByTagName('image')[0]
        img_width = img_info.getAttribute('width')
        img_height = img_info.getAttribute('height')
        return float(img_width), float(img_height)


class SVGParser_Vage:
    def __init__(self, filepath):
        self.dom = parse(filepath)
        self.root = self.dom.documentElement
        self.shapes = ['path', 'text', 'line', 'rect', 'circle']
        # vage-lite gradient need defs as fill 
        # filter the text element
        self.filtered_nodename = ['image', 'g', 'defs', 'linearGradient', 'stop' , 'circle', \
        'tspan', 'pattern', 'use', 'clipPath', 'svg', 'a', 'polyline'] # defs

    def _traverse_tree(self, root, ret_list, parent_attrs):
        parent_attrs = copy.copy(parent_attrs)
        
        if root.attributes is not None:
            attrs = root.attributes.items()
            for att in attrs:
                parent_attrs[att[0]] = att[1]

        for child in root.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                if child.nodeName in self.shapes:
                    attrs = child.attributes.items()
                    #print(attrs, child.nodeType)
                    attr_dict = copy.copy(parent_attrs)
                    for att in attrs:
                        attr_dict[att[0]] = att[1]
                    attr_dict['shape_name'] = child.nodeName
                    if child.childNodes:
                        # print(child.childNodes[0].data)
                        try:
                            attr_dict['data'] = child.childNodes[0].data
                        except:
                            attr_dict['data'] = "0"

                    # support tspan for plotly
                    if child.nodeName == "text":
                        for subchild in child.childNodes:
                            if subchild.nodeName == "tspan":
                                try:
                                    attr_dict['data'] = subchild.childNodes[0].data
                                except:
                                    attr_dict['data'] = "0"
                                attrs = subchild.attributes.items()
                                for att in attrs:
                                    attr_dict[att[0]] = att[1]
                                # print(attr_dict)
                    ret_list.append(attr_dict)
                elif child.nodeName not in self.filtered_nodename:
                    print('node is not a supported shape', child.tagName)
                    raise SystemExit
            self._traverse_tree(child, ret_list, parent_attrs)

    def get_all_shape(self):
        #obtain all shape
        ret_list = []
        self._traverse_tree(self.root, ret_list, {})
        return ret_list

    def get_image_size(self):
        # for Vage-lite, size in rect not in image
        try:
            img_info = self.root.getElementsByTagName('rect')[0]
            img_width = img_info.getAttribute('width')
            img_height = img_info.getAttribute('height')
        except:
            img_info = self.dom.getElementsByTagName('svg')[0]
            img_width = img_info.getAttribute('width')
            img_height = img_info.getAttribute('height')

        try:
            img_width, img_height = float(img_width), float(img_height)
        except:
            viewBox = img_info.getAttribute('viewBox')
            width = viewBox.split(" ")[2]
            height = viewBox.split(" ")[3]
            img_width, img_height = float(width), float(height)
        return img_width, img_height
