from svgpathtools import parse_path, wsvg
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc
from Datasets.a2c import a2c

class BezierParser:
    def shape2BezierPath(self, shape):
        if shape['shape_name'] == 'line':
            bezier_path = self.line2BezierPath(shape)
        elif shape['shape_name'] == 'path':
            bezier_path = self.path2BezierPath(shape)
        elif shape['shape_name'] == 'circle':
            bezier_path = self.circle2BezierPath(shape)
        else:
            print('svg shape commend not implemented:', shape['shape_name'])
            raise SystemExit
        return bezier_path
        
    def _a2c(self, arc_path):
        bezier_path = Path()
        x1 = arc_path.start.real
        y1 = arc_path.start.imag
        x2 = arc_path.end.real
        y2 = arc_path.end.imag
        rx = arc_path.radius.real
        ry = arc_path.radius.imag
        phi = arc_path.rotation
        fa = arc_path.large_arc
        fs = arc_path.sweep
        ret = a2c(x1, y1, x2, y2, fa, fs, rx, ry, phi)
        #print (ret)

        for i, curve in enumerate(ret):
            if i == 0:
                x1 = arc_path.start.real
                y1 = arc_path.start.imag
            else:
                x1 = curve[0].real
                y1 = curve[0].imag
            
            control0_x = curve[1].real
            control0_y = curve[1].imag
            control1_x = curve[2].real
            control1_y = curve[2].imag

            if i == len(ret) - 1:
                x2 = arc_path.end.real
                y2 = arc_path.end.imag
            else:
                x2 = curve[3].real
                y2 = curve[3].imag

            bezier_path.append(CubicBezier(complex(x1, y1), complex(control0_x, control0_y), 
                complex(control1_x, control1_y), complex(x2, y2)
            ))
        #print (bezier_path)
        return bezier_path

    def line2BezierPath(self, shape):
        #print(shape, 'fooo')
        bezier_path = Path()
        line = CubicBezier(
            complex(float(shape['x1']), float(shape['y1'])), 
            complex(float(shape['x1']), float(shape['y1'])), 
            complex(float(shape['x2']), float(shape['y2'])), 
            complex(float(shape['x2']), float(shape['y2']))
        )
        bezier_path.append(line)
        #node_dict = self._bezierPath2Graph(bezier_path, shape)
        
        #wsvg(line, filename = 'line_bezier.svg')
        #wsvg(Line(complex(float(shape['x1']), float(shape['y1'])), complex(float(shape['x2']), float(shape['y2']))), filename = 'line.svg')
        #print('fooo')
        return bezier_path

    def path2BezierPath(self, shape):
        #print(shape)
        path = parse_path(shape['d'])
        bezier_path = Path()
        for element in path:
            if isinstance(element, Arc):
                #wsvg(shape, filename = 'arc.svg')
                bezier_path += self._a2c(element)
                #wsvg(bezier_path, filename = 'bezier.svg')
            elif isinstance(element, Line):
                bezier_path += element
            else:
                print('shape not implemented in path commend:', element)
                raise SystemExit

        #node_dict = self._bezierPath2Graph(bezier_path, shape)
        
        return bezier_path
    
    def circle2BezierPath(self, shape):
        #print(shape)
        cx = float(shape['cx'])
        cy = float(shape['cy'])
        r = float(shape['r'])
        magic_number = r * 0.552284749831

        arc_top_right = CubicBezier(
            start = complex(cx, cy - r), 
            control1 = complex(cx + magic_number, cy - r), 
            control2 = complex(cx + r, cy - magic_number), 
            end = complex(cx + r, cy)
        )
        
        arc_bottom_right = CubicBezier(
            start = complex(cx + r, cy), 
            control1 = complex(cx + r, cy + magic_number), 
            control2 = complex(cx + magic_number, cy + r), 
            end = complex(cx, cy + r)
        )

        arc_bottom_left = CubicBezier(
            start = complex(cx, cy + r), 
            control1 = complex(cx - magic_number, cy + r), 
            control2 = complex(cx - r, cy + magic_number), 
            end = complex(cx - r, cy)
        )

        arc_top_left = CubicBezier(
            start = complex(cx - r, cy), 
            control1 = complex(cx - r, cy - magic_number), 
            control2 = complex(cx - magic_number, cy - r), 
            end = complex(cx, cy -r)
        )
        
        bezier_path = Path(arc_top_right, arc_bottom_right, 
            arc_bottom_left, arc_top_left
        )

        #wsvg(bezier_path, filename = 'circle_bezier.svg')
        #node_dict = self._bezierPath2Graph(bezier_path, shape)
        return bezier_path