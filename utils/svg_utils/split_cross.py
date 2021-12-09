from xml.dom.minidom import parse, Node, parseString
from svgpathtools import parse_path, wsvg
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
import copy
import math
import os

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

def merge_close_points(points):
    sim = euclidean_distances(points, points)
    sim = (sim < 1e-4)
    merged = np.zeros(sim.shape[0], dtype=bool)
    
    merged_points = []
    for i, s in enumerate(sim):
        candidates = points[(~merged) & s]
        if len(candidates) == 0: continue
        merged_points.append(np.mean(candidates, axis = 0))
        merged[s] = True

    return np.array(merged_points)

def split_circle(points, circles):
    circle_params = circles['param']
    if len(circle_params) == 0:
        return {'start_end':[], 'param':[], 'idx':[]},  circles
    cxs = circle_params[:, 0]
    cys = circle_params[:, 1]
    rs = circle_params[:, 2]
   
    def point_on_circle(x, y, cx, cy, r, th = 15):
        r2 = (x - cx) * (x - cx)  + (y - cy) * (y - cy)
        return np.abs(r2 - r * r) < th * th #r * r * th * th


    arc = {'start_end':[], 'param':[], 'idx':[]}
    un_splited_idx = []
    count = 0
    for circle_i, (cx, cy, r) in enumerate(zip(cxs, cys, rs)):
        on_curve = point_on_circle(points[:, 0], points[:, 1], cx, cy, r)
        #print(cx, cy, r, np.sum(on_curve))
        split_points = points[on_curve]
        if len(split_points) == 0: 
            un_splited_idx.append(circle_i)
            continue
        #print(split_points.shape)
        split_points = merge_close_points(split_points)
        

        def sort_points_by_angle(points, ascend = True):
            angle = np.arctan(points[:, 1] / points[:, 0])
            if ascend:
                idx = np.argsort(angle)
            else:
                idx = np.argsort(-angle)
            return idx

        if len(split_points) == 1:
            relative_pos = split_points - [cx, cy]
            #relative_pos = np.concatenate([relative_pos, -relative_pos])
            split_points = np.concatenate([split_points, [cx, cy] - relative_pos])

        #print('relative_pos', relative_pos)
        relative_pos = split_points - [cx, cy] + 1e-7
        
        #1st 4th Quadrant
        mask = (relative_pos[:, 0] > 0) & (relative_pos[:, 1] > 0)
        mask |= ((relative_pos[:, 0] > 0) & (relative_pos[:, 1] < 0))
        pos_4th_1st = relative_pos[mask]
        if len(pos_4th_1st) != 0:
            idx = sort_points_by_angle(pos_4th_1st)
            pos_4th_1st = split_points[mask][idx]
        else:
            pos_4th_1st = np.zeros((0, 2))
        
        #print('1st/4th quadrant', pos_4th_1st - [cx, cy])

        #2nd Quadrant
        mask = (relative_pos[:, 0] < 0) & (relative_pos[:, 1] > 0)
        pos_2nd = relative_pos[mask]
        if len(pos_2nd) !=0:
            idx = sort_points_by_angle(pos_2nd)
            pos_2nd = split_points[mask][idx]
        else:
            pos_2nd = np.zeros((0, 2))
        #print('2nd quadrant', pos_2nd - [cx, cy])

        #3rd Quadrant
        mask = (relative_pos[:, 0] < 0) & (relative_pos[:, 1] < 0)
        pos_3rd = relative_pos[mask]
        if len(pos_3rd) != 0:
            idx = sort_points_by_angle(pos_3rd)
            pos_3rd = split_points[mask][idx]
        else:
            pos_3rd = np.zeros((0, 2))
        #print('3rd quadrant', pos_3rd - [cx, cy])

        
        
        sorted_pos = np.concatenate([pos_4th_1st, pos_2nd, pos_3rd], axis = 0)
        #print('sorted', sorted_pos ,sorted_pos - [cx, cy])
            

        def build_arc(start, end, cx, cy, r):
            x0 = start[0]
            y0 = start[1]
            x1 = end[0]
            y1 = end[1]
            rx = r
            ry = r
            rot = 0
            o = [cx, cy]
            
            start_vector = start - o
            end_vector = end - o
            
            a = start_vector[1] / (start_vector[0] + 1e-7) #slope of the line cross starting point
            if start_vector[0] > 0: #1st/4th quadrant
                if end_vector[1] > a * end_vector[0]: #arc blow the line
                    large_arc = 0
                else: #arc above the line
                    large_arc = 1
            else:
                if end_vector[1] > a * end_vector[0]:
                    large_arc = 1
                else:  
                    large_arc = 0
            #print(start, end, start_vector, end_vector, a, a * end_vector[0], large_arc, 'foooo')
            sweep = 1
            start_end = [x0, y0, x1, y1]
            param = [rx, ry, rot, large_arc, sweep]
            return start_end, param

        for i in range(0, len(sorted_pos) - 1):
            start = sorted_pos[i]
            end = sorted_pos[i + 1]
            
            start_end, param = build_arc(start, end, cx, cy, r)
            arc['start_end'].append(start_end)
            arc['param'].append(param)
            #arc['idx'].append(count)
            count += 1

        start_end, param = build_arc(sorted_pos[-1], sorted_pos[0], cx, cy, r)
        arc['start_end'].append(start_end)
        arc['param'].append(param)
        #arc['idx'].append(count)
        count += 1
        
    circles = {'param':circles['param'][un_splited_idx]}
    for key in arc:
        arc[key] = np.array(arc[key])
    
        
    if False:
        paths = []
        for start_end, param in zip(arc['start_end'], arc['param']):
            print(start_end, param)
            path = Path()
            start = complex(start_end[0], start_end[1])
            radius = complex(param[0], param[1])
            rotation = param[2]
            large_arc = param[3]
            sweep = param[4]
            end = complex(start_end[2], start_end[3])
            path.append(Arc(start, radius, rotation, large_arc, sweep, end))
            paths.append(path)
        print(paths)
        wsvg(paths, filename = 'debug.svg')

    
    return arc, circles
        
'''
def split_arc(points, arcs):
    for arc_i in range(len(arc['start_end'])):
        a_x0 = arc['start_end'][arc_i, 0]
        a_y0 = arc['start_end'][arc_i, 1]
        a_x1 = arc['start_end'][arc_i, 2]
        a_y1 = arc['start_end'][arc_i, 3]
        rx = arc['param'][arc_i, 0]
        ry = arc['param'][arc_i, 1]
        large_arc = arc['param'][arc_i, 2]
        sweep = arc['param'][arc_i, 3]
        
        on_curve = point_on_arc(points[:, 0], points[:, 1], a_x0, a_y0, a_x1, a_y1, rx, ry, large_arc, sweep)
'''

def split_line(points, lines):
    def point_on_line(x, y, x0, y0, x1, y1, th = 3): 
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        out_rect = (x - min_x < -1) | (x - max_x > 1) | (y - min_y < -1) | (y - max_y > 1)
        is_start_end = (x - min_x <= 1) & (x - min_x >= -1) & (y - min_y <= 1) & (y - min_y >= -1)
        is_start_end |= (x - max_x <= 1) & (x - max_x >= -1) & (y - max_y <= 1) & (y - max_y >= -1)
        valid = ~is_start_end

        if x1 - x0 != 0:
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            
            d_p2l_2 = (a * x - y + b) * (a * x - y + b) / (a * a + 1)
            x_proj = (a * (y - b) + x) / (a * a + 1)
            y_proj = a * x_proj + b
            
        else:
            d_p2l_2 = (x - x0) * (x - x0)
            x_proj = x0
            y_proj = y

        close_to_line = d_p2l_2 < th * th #((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0)) * th * th
        within_start_end = (x_proj >= min_x) & (x_proj <= max_x) & (y_proj >= min_y) & (y_proj <= max_y)
        
        '''
        if min_x == 258.979986 and min_y == 405.309139:
            print(close_to_line)
            for i, p in enumerate(points):
                print(i, p)
            print(close_to_line[31], d_p2l_2[31], ((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0)) * th * th)
            print(valid[31], within_start_end[31])
            print((valid & close_to_line & within_start_end)[31], points[31], [x0, y0, x1, y1])
            raise SystemExit
        '''
        
        return valid & close_to_line & within_start_end

    new_lines = {'start_end':[]}
    for line_i in range(len(lines['start_end'])):
        line_x0 = lines['start_end'][line_i, 0]
        line_y0 = lines['start_end'][line_i, 1]
        line_x1 = lines['start_end'][line_i, 2]
        line_y1 = lines['start_end'][line_i, 3]
        

        on_curve = point_on_line(points[:, 0], points[:, 1], line_x0, line_y0, line_x1, line_y1)

        split_points = points[on_curve]

        if len(split_points) == 0: 
            new_lines['start_end'].append(lines['start_end'][line_i])
            continue
        #print(split_points.shape)
        split_points = merge_close_points(split_points)
        split_points = np.concatenate([np.array([line_x0, line_y0])[None, :], split_points, np.array([line_x1, line_y1])[None, :]])

        if line_x1 == line_x0:
            idx = np.argsort(split_points[:, 1])
            split_points = split_points[idx]
        else:
            a = (line_y1 - line_y0) / (line_x1 - line_x0)
            if np.abs(a) > 0.5:
                idx = np.argsort(split_points[:, 1])
                split_points = split_points[idx]
            else:
                idx = np.argsort(split_points[:, 0])
                split_points = split_points[idx]

        '''
        if lines['start_end'][line_i][0] == 258.979986 and lines['start_end'][line_i][1] == 405.309139 and lines['start_end'][line_i][2] == 322.173958 and lines['start_end'][line_i][3] == 405.309139:
            print(split_points)
            raise SystemExit
        '''
        

        #print(split_points)
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            new_lines['start_end'].append(np.concatenate([start, end]))
        #print(new_lines)
    return new_lines

def split_cross(shape_list):
    #print(shape_list)
    start_end = []
    type_dict = {'line':{'start_end':[]}, #, 'idx':[]},
        'circle':{'param':[]}, #, 'idx':[]}, 
        'arc':{'start_end':[], 'param':[]} #, 'idx':[]}
    }

    for i, shape in enumerate(shape_list):
        if shape['shape_name'] == 'line':
            x0 = float(shape['x1'])
            y0 = float(shape['y1'])
            x1 = float(shape['x2'])
            y1 = float(shape['y2'])

            type_dict['line']['start_end'].append([x0, y0, x1, y1])
            #type_dict['line']['idx'].append(i)
        elif shape['shape_name'] == 'circle':
            cx = float(shape['cx'])
            cy = float(shape['cy'])
            r = float(shape['r'])
            type_dict['circle']['param'].append([cx, cy, r])
            #type_dict['circle']['idx'].append(i)
        elif shape['shape_name'] == 'path':
            cmd = shape['d']
            path = parse_path(cmd)
            if len(path)> 1:
                print('Error: Un-implemented path length')
            arc = path[0]
            if not isinstance(arc, Arc):
                print('Error: Un-implemented path type')
            x0 = arc.start.real
            y0 = arc.start.imag
            x1 = arc.end.real
            y1 = arc.end.imag
            rx = arc.radius.real
            ry = arc.radius.imag
            rot = arc.rotation
            large_arc = arc.large_arc
            sweep = arc.sweep
            type_dict['arc']['start_end'].append([x0, y0, x1, y1])
            type_dict['arc']['param'].append([rx, ry, rot, large_arc, sweep])
            #type_dict['arc']['idx'].append(i)
            
        else:
            print('Error: Un-implemented shape', shape)
            raise SystemExit
    
    for shape_type in type_dict:
        for key in type_dict[shape_type]:
            type_dict[shape_type][key] = np.array(type_dict[shape_type][key])

    
    arc, unsplited_circle = split_circle(type_dict['line']['start_end'].reshape((-1, 2)), type_dict['circle'])
    #split_arc(type_dict['line']['start_end'].reshape((-1, 2)), type_dict['arc'])
    type_dict['line'] = split_line(type_dict['line']['start_end'].reshape((-1, 2)), type_dict['line'])
    
    type_dict['circle'] = unsplited_circle
    for key in type_dict['arc']:
        if len(arc[key]) == 0: continue
        if len(type_dict['arc'][key]) == 0:
            type_dict['arc'][key] = arc[key]
        else:
            type_dict['arc'][key] = np.concatenate([type_dict['arc'][key], arc[key]], axis = 0)      
        
            
    return type_dict

if __name__ == '__main__':
    input_dir = '/data/xinyangjiang/Datasets/SESYD/diagram2'
    output_dir = '../DiagramSplitCross'
    dir_list = os.listdir(input_dir)
    for dir_name in dir_list:
        if not os.path.isdir(os.path.join(input_dir, dir_name)):
            continue
        svg_list = os.listdir(os.path.join(input_dir, dir_name))
        for svg_name in svg_list:
            #if dir_name != 'floorplans16-06' or svg_name != 'file_13.svg':continue
            if '.svg' not in svg_name: continue
            print(os.path.join(dir_name, svg_name))
            p = SVGParser(os.path.join(input_dir, dir_name, svg_name))
            type_dict = split_cross(p.get_all_shape())

            paths = []
            for start_end in type_dict['line']['start_end']:
                path = Path(Line(
                    start = complex(start_end[0], start_end[1]), 
                    end = complex(start_end[2], start_end[3]), 
                ))
                paths.append(path)
            
            
            for start_end, param in zip(type_dict['arc']['start_end'], type_dict['arc']['param']):
                start = complex(start_end[0], start_end[1])
                radius = complex(param[0], param[1])
                rotation = param[2]
                large_arc = param[3]
                sweep = param[4]
                end = complex(start_end[2], start_end[3])
                path = Path(Arc(start, radius, rotation, large_arc, sweep, end))
                paths.append(path)
            
            for param in type_dict['circle']['param']:
                cx, cy, r = param
                start = complex(cx - r, cy)
                end = complex(cx + r, cy)
                radius = complex(r, r)
                rotation = 0
                sweep = 0
                path = Path(Arc(start, radius, rotation, 0, 0, end))
                paths.append(path)
                path = Path(Arc(start, radius, rotation, 0, 1, end))
                paths.append(path)
            
            if not os.path.isdir(os.path.join(output_dir, dir_name)):
                os.mkdir(os.path.join(output_dir, dir_name))
            wsvg(paths, filename = os.path.join(output_dir, dir_name, svg_name))
