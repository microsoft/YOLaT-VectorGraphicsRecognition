import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import numpy as np
import pickle
import math
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from svgpathtools import parse_path, wsvg
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc

from Datasets.svg_parser import SVGGraphBuilderBezier2 as SVGGraphBuilderBezier
from Datasets.svg_parser import SVGParser
from Datasets.bezier_parser import BezierParser

from utils.svg_utils.split_cross import split_cross

def shape2Path(type_dict):
    parser = BezierParser()
    paths = Path()
    if 'line' in type_dict:
      for start_end in type_dict['line']['start_end']:
          x0, y0, x1, y1 = start_end
          path = parser.line2BezierPath({'x1':x0, 'y1':y0, 'x2':x1, 'y2':y1})
          paths += path
    if 'arc' in type_dict:
      for start_end, param in zip(type_dict['arc']['start_end'], type_dict['arc']['param']):
          start = complex(start_end[0], start_end[1])
          radius = complex(param[0], param[1])
          rotation = param[2]
          large_arc = param[3]
          sweep = param[4]
          end = complex(start_end[2], start_end[3])
          #path = Path(Arc(start, radius, rotation, large_arc, sweep, end))
          path = parser._a2c(Arc(start, radius, rotation, large_arc, sweep, end))
          paths += path
        
    if 'circle' in type_dict:
      for param in type_dict['circle']['param']:
          cx, cy, r = param
          path = parser.circle2BezierPath({'cx':cx, 'cy':cy, 'r':r})
          paths += path

    return paths

def getConnnectedComponent(node_dict):
    edges = node_dict['edge']['shape']
    pos = node_dict['pos']['spatial']
    is_control = node_dict['attr']['is_control']
    #print(edges)
    adj = np.eye(pos.shape[0], pos.shape[0]).astype(np.bool)
    for e in edges:
        adj[e[0], e[1]] = True
        adj[e[1], e[0]] = True

    n_node = pos.shape[0]
    visited = [False if not is_control[i] else True for i in range(n_node) ]
    clusters = []

    for start_node in range(0, n_node):
        if visited[start_node]: continue
        
        cluster = [start_node]
        visited[start_node] = True
        queue = [start_node]

        while len(queue) != 0:
            node_idx = queue.pop(0)
            neighbors = adj[node_idx]
            for i in range(0, n_node):
                if neighbors[i] and not visited[i]:
                    cluster.append(i)
                    visited[i] = True
                    queue.append(i)
    
        clusters.append(cluster) 

    return clusters    

def draw_cluster_graph(svg_path, save_path, width, height, bboxs, pos, is_control, edges):
    tiff_path = svg_path.replace('.svg', '.tiff') 
    img = np.array(Image.open(tiff_path))
    fig, ax = plt.subplots(ncols=1)
    ax.imshow(img)

    for idx, (node, control) in enumerate(zip(pos, is_control)):
      if control:
        ax.text(node[0] * width, node[1] * height, s=idx, color='red', fontsize=1)
      else:
        ax.text(node[0] * width, node[1] * height, s=idx, color='black', fontsize=2)
    for node1, node2 in edges:
      node1_x, node1_y = pos[node1]
      node2_x, node2_y = pos[node2]
      ax.plot( [node1_x * width, node2_x * width] , [node1_y * height, node2_y * height], linewidth=1 )

    for proposal in bboxs:
      bbox = patches.Rectangle(
            (proposal[0] * width, proposal[1] * height),
            proposal[2] * width - proposal[0] * width,
            proposal[3] * height - proposal[1] * height,
            linewidth=0.3,
            edgecolor="red",
            facecolor="none",
        )
      ax.add_patch(bbox)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0, dpi=600)
 
def mergeCluster(cc, bboxs, ratio=None, expand_length=None):
  # default: ratio:0.5, expand_length:0.06
  from collections import deque
  from pathlib import Path
  utils_dir = (Path(__file__).parent.parent.parent).resolve()
  if utils_dir not in sys.path: sys.path.insert(0, str(utils_dir))
  from det_util import bbox_iou
  # expand how much ratio
  def expand_bboxs(bboxs):
    expand_bboxs = []
    for box in bboxs:
      min_x, min_y, max_x, max_y = box
      x_len , y_len = (max_x - min_x), (max_y - min_y)
      if not expand_length:
        delta_min_x, delta_min_y, delta_max_x, delta_max_y = - ratio*x_len, - ratio*y_len, ratio*x_len, ratio*y_len
      elif not ratio:
        delta_min_x, delta_min_y, delta_max_x, delta_max_y = - expand_length[0], -expand_length[1], expand_length[0], expand_length[1]
      else: raise NotImplementedError('invalid expand')
      expand_min_x = max(0, min_x + delta_min_x)
      expand_min_y = max(0, min_y + delta_min_y)
      expand_max_x = min(1, max_x + delta_max_x)
      expand_max_y = min(1, max_y + delta_max_y)
      expand_bboxs.append([expand_min_x, expand_min_y, expand_max_x, expand_max_y])
    return expand_bboxs

  expand_bboxs = expand_bboxs(bboxs)
  expand_bboxs = np.array(expand_bboxs)
  adj_matrix = np.zeros([len(cc), len(cc)])
  for idx, box in enumerate(expand_bboxs):
    max_over_min_x = np.where( box[0] > expand_bboxs[:, 0], box[0], expand_bboxs[:, 0])
    min_over_max_x = np.where( box[2] > expand_bboxs[:, 2], expand_bboxs[:, 2], box[2])
    max_over_min_y = np.where( box[1] > expand_bboxs[:, 1], box[1], expand_bboxs[:, 1])
    min_over_max_y = np.where( box[3] > expand_bboxs[:, 3], expand_bboxs[:, 3], box[3])
    adj_matrix[idx] = np.logical_and(max_over_min_x <= min_over_max_x, max_over_min_y <= min_over_max_y)
  
  # idx of the nodes
  new_cc = []
  # idx of the cc
  cc_merged = []
  visited = [False] * len(cc)

  for c in range(len(cc)):
    if visited[c]: continue
    c_merged = [c]
    new_c = cc[c]
    visited[c] = True
    current_cluster = deque([c])
    while current_cluster:
      node = current_cluster.popleft()
      for node_idx, has_edge in enumerate(adj_matrix[node]):
        if has_edge and not visited[node_idx]:
          c_merged.append(node_idx)
          new_c.extend(cc[node_idx])
          current_cluster.append(node_idx)
          visited[node_idx] = True
    cc_merged.append(c_merged)
    new_cc.append(new_c)
    

  new_bboxs = []
  for cc_m in cc_merged:
    cc_bboxs = []
    for c in cc_m:
      cc_bboxs.append(bboxs[c]) 
    cc_bboxs = np.array(cc_bboxs)
    new_bboxs.append([min(cc_bboxs[:, 0]), min(cc_bboxs[:, 1]), max(cc_bboxs[:, 2]), max(cc_bboxs[:, 3])])
  return new_cc, new_bboxs  

    #draw_cluster_graph("/home/v-luliu1/datasets/diagram/diagrams21-07/file_73.svg", "/home/v-luliu1/datasets/73.pdf",4197.856863, 1870.7141609999999, bboxs, pos, is_control, edges)
def mergeCC(node_dict, svg_path, width, height):
    edges = node_dict['edge']['shape']
    pos = node_dict['pos']['spatial']
    color = node_dict['attr']['color']
    is_control = node_dict['attr']['is_control']

    cc = getConnnectedComponent(node_dict)

    bboxs = []
    for i, cluster in enumerate(cc):
        pos_cluster = pos[cluster]
        max_x = pos_cluster[:, 0].max(0)
        min_x = pos_cluster[:, 0].min(0)
        max_y = pos_cluster[:, 1].max(0)
        min_y = pos_cluster[:, 1].min(0)
        bboxs.append((min_x, min_y, max_x, max_y))

    #cc, bboxs = mergeCluster(cc, bboxs, ratio=0.5, expand_length=None)
    # hardcode 70 for expand_length
    #cc, bboxs = mergeCluster(cc, bboxs, ratio=None, expand_length=(70 / width, 70 / height))
    cc, bboxs = mergeCluster(cc, bboxs, ratio=None, expand_length=(40 / width, 40 / height))

    paths = []
    shape_shape_edges = []
    for i, cluster in enumerate(cc):
        for idx in cluster:
            for idx_j in cluster:
                if idx == idx_j: continue
                shape_shape_edges.append((idx, idx_j))

        if True:
            real_max_x = pos_cluster[:, 0].max(0) * width
            real_min_x = pos_cluster[:, 0].min(0) * width
            real_max_y = pos_cluster[:, 1].max(0) * height
            real_min_y = pos_cluster[:, 1].min(0) * height
            
            p0 = complex(real_min_x, real_min_y)
            p1 = complex(real_max_x, real_min_y)
            p2 = complex(real_max_x, real_max_y)
            p3 = complex(real_min_x, real_max_y)
            paths.append(Path(Line(p0, p1), 
                Line(p1, p2), 
                Line(p2, p3), 
                Line(p3, p0)
            ))

    svg_path_list = svg_path.split('/')
    
    #draw_cluster_graph(svg_path, "/home/v-luliu1/datasets/diagram_expand_len_graph/{}_{}".format(svg_path_list[-2], svg_path_list[-1].replace('.svg', '.pdf')), width, height, bboxs, pos, is_control, edges)
    #print("draw bbox and node of {}".format(svg_path))

    cross_shape_edges = []
    same_cc = np.zeros((len(bboxs), len(bboxs))).astype(np.bool)
    for i, parent_bb in enumerate(bboxs):
        for j, child_bb in enumerate(bboxs):
            if i == j: continue
            inter_rect_x1 = max(parent_bb[0], child_bb[0])
            inter_rect_y1 = max(parent_bb[1], child_bb[1])
            inter_rect_x2 = min(parent_bb[2], child_bb[2])
            inter_rect_y2 = min(parent_bb[3], child_bb[3])

            child_area = (child_bb[2] - child_bb[0]) * (child_bb[3] - child_bb[1])
            is_parent_child = False
            
            if child_area > 0:
                inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * max(inter_rect_y2 - inter_rect_y1, 0)
                if inter_area * 1.0 / child_area > 0.9:
                    is_parent_child = True
            else:
                if child_bb[2] - child_bb[0] == 0:
                    if inter_rect_x2 - inter_rect_x1 == 0 and max(inter_rect_y2 - inter_rect_y1, 0) > 0.9 * (child_bb[3] - child_bb[1]):
                        is_parent_child = True
                if child_bb[3] - child_bb[1] == 0:
                    if  max(inter_rect_x2 - inter_rect_x1, 0) > 0.9 * (child_bb[2] - child_bb[0]) and inter_rect_y2 - inter_rect_y1 == 0:
                        is_parent_child = True
            
            if is_parent_child:
                for parent_idx in cc[i]:
                    for child_idx in cc[j]:
                        cross_shape_edges.append((parent_idx, child_idx))
                        same_cc[i, j] = True
                        same_cc[j, i] = True
    

 
    def get_all_neighboors(root, ret):
        all_neighbors = same_cc[root]
        for i, is_neighbor in enumerate(all_neighbors):
            if i == root: continue
            if not is_neighbor: continue
            if visited[i]: continue
            ret.append(i)
            visited[i] = True
            get_all_neighboors(i, ret)

    visited = np.zeros(same_cc.shape[0]).astype(np.bool)
    merged_cc = []
    for i, all_neighbors in enumerate(same_cc):
        if visited[i]: continue
        cluster = [i]
        get_all_neighboors(i, cluster)
        merged_cc.append(cluster)
        visited[i] = True

    new_cc = []
    for cluster in merged_cc:
        t = []
        for idx in cluster:
            t += cc[idx]
        new_cc.append(t)
    
    def get_attr(edges): 
        ret = []
        for e in edges:
            pos_start = pos[e[0]]
            pos_end = pos[e[1]]

            euc_d2 = (pos_start[0] - pos_end[0]) * (pos_start[0] - pos_end[0]) + (pos_start[1] - pos_end[1]) * (pos_start[1] - pos_end[1])
            angle = (pos_start[0] - pos_end[0]) / (np.sqrt(euc_d2) + 1e-7)
            w = 1 / np.exp(euc_d2)
            #w = (w - 0.8) / 0.2
            #print(angle, w)
            if math.isnan(angle):
                print(angle, pos_start, pos_end, dot_prod, cos_theta)
                raise SystemExit

            ret.append([0, 0, 0, 0, angle, euc_d2])
        return ret

    shape_shape_edge_attr = get_attr(shape_shape_edges)
    cross_shape_edge_attr = get_attr(cross_shape_edges)
    
    return np.array(shape_shape_edges), np.array(cross_shape_edges), np.array(shape_shape_edge_attr), np.array(cross_shape_edge_attr), paths, new_cc



if __name__ == '__main__':
    graph_builder = SVGGraphBuilderBezier()
    #input_dir = '/home/v-luliu1/datasets/floorplans_test'
    #output_dir = '/home/v-luliu1/datasets/floorplans_test'
    input_dir = '/data/xinyangjiang/Datasets/SESYD/diagram2'
    output_dir = '/data/xinyangjiang/Datasets/SESYD/diagram2'
    dir_list = os.listdir(input_dir)

    angles = []
    distances = []
    for dir_name in dir_list:
        if not os.path.isdir(os.path.join(input_dir, dir_name)):
            continue
        svg_list = os.listdir(os.path.join(input_dir, dir_name))
        for svg_name in svg_list:
            if '.svg' not in svg_name: continue
            filepath = os.path.join(input_dir, dir_name, svg_name)
            print(filepath)
            p = SVGParser(filepath)
            # split_cross splits the segments into multiple small segments if there is a cross-point
            type_dict = split_cross(p.get_all_shape())
            width, height = p.get_image_size()
            paths = shape2Path(type_dict)
            # {'pos': {'spatial': the positions of nodes}, {'attr': {'color': color of every node, 'stroke_width': stroke width, 'is_control': if the node is control node}}, {'edge': {'control': control edge, 'spatial': spatial edge}}, 'edge_attr': N * 6}
            # edge is between any two non-control nodes in a bezier curve
            # the intersection nodes are added as nodes
            # What does the N in edge_attr mean?
            node_dict = graph_builder.bezierPath2Graph(paths, 
                {'width':width, 
                'height':height, 
                'stroke':'black', 
                'stroke-width': 6}
            )
            
            #print(node_dict['edge']['shape'])
            for key in node_dict:
                for k in node_dict[key]:
                    node_dict[key][k] = np.array(node_dict[key][k])
                    if len(node_dict[key][k].shape) == 1:
                        node_dict[key][k] = node_dict[key][k][:, None]
                    #print(key, k, node_dict[key][k].shape)

            node_dict = graph_builder.mergeNode(node_dict)
            if True:
                e = node_dict['edge']['shape']
                for ee in e:
                    if ee[0] == ee[1]:
                        print(ee)
            
            #getConnnectedComponent(node_dict)
            #super_pos, super_color, shape_shape_edges, super_shape_edges, super_super_edges, bbox_paths = getSuperNode(node_dict)
            
            shape_shape_edges, cross_shape_edges, shape_shape_edge_attr, cross_shape_edge_attr, bbox_paths, cc = mergeCC(node_dict, filepath, width, height)

            bbox_paths.append(paths)
            
            start_end_size = node_dict['pos']['spatial'].shape[0]
            #node_dict['pos']['spatial'] = np.concatenate([node_dict['pos']['spatial'], super_pos], axis = 0)
            #node_dict['attr']['color'] = np.concatenate([node_dict['attr']['color'], super_color], axis = 0)
            #node_dict['edge']['super'] = np.concatenate([shape_shape_edges, super_shape_edges, super_super_edges], axis = 0)
            # fix bug: if there's no cross shape edges
            if len(cross_shape_edges) == 0:
              node_dict['edge']['super'] = shape_shape_edges
            else:  
              node_dict['edge']['super'] = np.concatenate([shape_shape_edges, cross_shape_edges], axis = 0)
            #node_dict['attr']['is_control'] = np.concatenate([node_dict['attr']['is_control'], np.zeros((super_pos.shape[0], 1)).astype(np.bool)], axis = 0)
            #node_dict['attr']['is_super'] = np.concatenate([np.zeros((start_end_size, 1)).astype(np.bool), np.ones((super_pos.shape[0], 1)).astype(np.bool)], axis = 0)
            node_dict['attr']['is_super'] = np.zeros((start_end_size, 1)).astype(np.bool)
            if len(cross_shape_edge_attr) == 0:
              node_dict['edge_attr']['super'] = shape_shape_edge_attr
            else:
              node_dict['edge_attr']['super'] = np.concatenate([shape_shape_edge_attr, cross_shape_edge_attr], axis = 0)
            #print(node_dict['attr']['is_control'].shape, node_dict['attr']['is_super'].shape)
            node_dict['img_width'] = width
            node_dict['img_height'] = height
            node_dict['cc'] = cc
            
            if not os.path.isdir(os.path.join(output_dir, dir_name)):
                os.mkdir(os.path.join(output_dir, dir_name))
            #wsvg(bbox_paths, filename = os.path.join(output_dir, dir_name, svg_name))
            output_name = svg_name.replace('.svg', '.pkl')
            pickle.dump(node_dict, open(os.path.join(output_dir, dir_name, output_name), 'wb'))

            for a in node_dict['edge_attr']['super']:
                angles.append(a[4])
                distances.append(a[5])
            
    
    stats = {'angles':{'mean':np.mean(angles), 'std':np.std(angles)}, 
        'distances':{'mean':np.mean(distances), 'std':np.std(distances)}
    }
    pickle.dump(stats, open(os.path.join(output_dir, 'stats.pkl'), 'wb'))
    print(stats)
