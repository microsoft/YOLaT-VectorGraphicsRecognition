import torch
from torch.nn import Linear as Lin
import torch_geometric as tg
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, PlainDynBlock, ResBlock, DenseDynBlock, DilatedKnnGraph
from torch_scatter import scatter
from torch_geometric.data import Data
import numpy as np
import time
from thop import profile
from fvcore.nn import FlopCountAnalysis

class Backbone(torch.nn.Module):
    def __init__(self, opt, n_edges = 3, edge_max_pool = torch.nn.AdaptiveAvgPool1d):
        super(Backbone, self).__init__()
        channels = opt.n_filters
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        conv = 'attr_edge_gp2' #opt.conv
        c_growth = channels
        n_edges = 1
        self.n_edges = n_edges

        self.n_blocks = opt.n_blocks
        self.n_blocks_out = opt.n_blocks_out
        self.heads = torch.nn.ModuleList()
        self.n_classes = opt.n_classes
        self.class_specific = opt.class_specific

        
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias)

        self.backbone = MultiSeq(*[ResBlock(channels, conv, act, norm, bias)
                                   for i in range(self.n_blocks-1)])
        fusion_dims = int(channels + c_growth * (self.n_blocks_out - 1))

        self.fusion_block = MLP([fusion_dims, 1024], act, norm, bias)
        self.fusion_block_super = MLP([fusion_dims, 1024], act, norm, bias)
        self.fusion_dims = fusion_dims

    def forward(self, x, edges, edge_weights, edge_attrs, bbox_idx):       
        f, f_super = self.head(x, edges[0], edge_weights[0], edge_attrs[0], x_node = x)

        feats = [f]
        feats_super = [f_super]

        for i in range(self.n_blocks-1):
            f = feats[-1]
            f_super = feats_super[-1]

            
            f, f_super = self.backbone[i](f, edges[0], edge_weights[0], edge_attrs[0], x_node = f_super)
            feats.append(f)
            feats_super.append(f_super)
            
        
        feats = [feats[i] for i in range(self.n_blocks - self.n_blocks_out, self.n_blocks)]
        feats = torch.cat(feats, dim=1)
        fusion_feats = self.fusion_block(feats)
        out_feat = torch.cat((fusion_feats, feats), dim = 1)
        
        feats_super = [feats_super[i] for i in range(self.n_blocks - self.n_blocks_out, self.n_blocks)]
        feats_super = torch.cat(feats_super, dim=1)
        feats_super = scatter(feats_super, bbox_idx, dim = 0, reduce = 'mean')
        fusion_feats_super = self.fusion_block_super(feats_super)
        out_feat_super = torch.cat((fusion_feats_super, feats_super), dim = 1)

        return out_feat, out_feat_super
    
class SparseCADGCN(torch.nn.Module):
    def __init__(self, opt, n_edges = 3, edge_max_pool = torch.nn.AdaptiveAvgPool1d, expand_ratio = 0.25):
        super(SparseCADGCN, self).__init__()

        self.expand_ratio = expand_ratio
        channels = opt.n_filters
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        conv = opt.conv
        c_growth = channels
        self.n_classes = opt.n_classes
        self.classifier = opt.classifier
        self.class_specific = opt.class_specific
        self.dim_stat = 0 # 13 #16
                
        #self.regress_net = Backbone(opt)
        self.cls_net = Backbone(opt)
        self.prediction_cls = MultiSeq(*[MLP([(self.cls_net.fusion_dims+ 1024) * 2 + self.dim_stat, 512], act, norm, bias),
            MLP([512, 256], act, norm, bias, drop=opt.dropout),
            MLP([256, opt.n_classes], None, None, bias)])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data, slices):
        x = data.x.cuda()
        bbox_idx = data.bbox_idx.cuda()

        edges = [data.edge.cuda().T]
        pred_bbox = data.bbox.cuda()
        stat_feats = data.stat_feats.cuda()

        edge_weights = [None]
        edge_attrs = [data.e_attr.cuda()]

        #print("--- overhead %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()

        #out_feat_regress = self.regress_net(x, edges, edge_weights, edge_attrs)
        out_feat_cls, out_feat_cls_super = self.cls_net(x, edges, edge_weights, edge_attrs, bbox_idx)
        out_feat_cls = scatter(out_feat_cls, bbox_idx, dim = 0, reduce = 'max')

        #print("--- feature extract %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()

        out_feat_cls = torch.cat([out_feat_cls, out_feat_cls_super], dim = 1)
        pred_cls = self.prediction_cls(out_feat_cls)

        #print("---classification %s seconds ---" % (time.time() - start_time))
        
        if self.classifier != 'softmax':
            pred_cls = torch.sigmoid(pred_cls)
        else:
            pred_cls = pred_cls

        return pred_cls, pred_bbox
    
    def predict(self, data, slices):
        roots = data.roots
        #print(slices)

        slice_pos = []
        slice_edge = []
        slice_edge_super = []
        slice_bbox = []

        slice_root = slices['roots']
        slice_image_bbox_root = [0]

        #torch.cuda.synchronize() 
        #st = time.time()
        for i in range(0, len(slice_root) - 1):
            start = slice_root[i]
            end = slice_root[i + 1]
            sub_roots = roots[start:end]
            for root in sub_roots:
                slice_pos += list(range(root.value['idx_pos'][0] + slices['pos'][i], root.value['idx_pos'][1] + slices['pos'][i]))
                slice_edge += list(range(root.value['idx_edge'][0] + slices['edge'][i], root.value['idx_edge'][1] + slices['edge'][i]))
                slice_edge_super += list(range(root.value['idx_edge_super'][0] + slices['edge_super'][i], root.value['idx_edge_super'][1] + slices['edge_super'][i]))
                slice_bbox.append(int(root.value['idx_bbox'] + slices['bbox'][i]))
            slice_image_bbox_root.append(len(slice_bbox))
        #torch.cuda.synchronize() 
        #et = time.time()
        #print('obtain slice 0: ', et-st)

        def build_data(data, slice_pos, slice_edge, slice_edge_super, slice_bbox):
            #torch.cuda.synchronize() 
            #st = time.time()
            o2n = {}
            count = 0
            for new_i, old_i in enumerate(slice_pos):
                o2n[old_i] = new_i
            #torch.cuda.synchronize() 
            #et = time.time()
            #print('o2n: ', et-st)
            
            #torch.cuda.synchronize() 
            #st = time.time()
            new_data = Data(x = data.x[slice_pos], pos = data.pos[slice_pos])
            new_data.bbox_idx = data.bbox_idx[slice_pos]
            #torch.cuda.synchronize() 
            #et = time.time()
            #print('data init: ', et-st)
            
            #torch.cuda.synchronize() 
            #st = time.time()
            new_data.edge = []
            for e in data.edge[slice_edge].numpy():
                new_data.edge.append([o2n[e[0]], o2n[e[1]]])
            new_data.edge = torch.tensor(new_data.edge, dtype = torch.long)
            #torch.cuda.synchronize() 
            #et = time.time()
            #print('edge init: ', et-st)

            '''
            torch.cuda.synchronize() 
            st = time.time()
            new_data.edge_super = []
            for e in data.edge_super[slice_edge_super].numpy():
                new_data.edge_super.append([o2n[e[0]], o2n[e[1]]])
            torch.cuda.synchronize() 
            et = time.time()
            print('edge generate: ', et-st)
            torch.cuda.synchronize() 
            st = time.time()
            new_data.edge_super = torch.tensor(new_data.edge_super, dtype = torch.long)
            torch.cuda.synchronize() 
            et = time.time()
            print('edge on gpu: ', et-st)
            '''
    
            #torch.cuda.synchronize() 
            #st = time.time()
            new_data.e_attr = data.e_attr[slice_edge]
            #new_data.e_attr_super = data.e_attr_super[slice_edge_super]
            new_data.bbox = data.bbox[slice_bbox]
            new_data.stat_feats = data.stat_feats[slice_bbox]
            #torch.cuda.synchronize() 
            #et = time.time()
            #print('slice other: ', et-st)
            #print(new_data.x.size(), new_data.bbox_idx.size(),  len(sorted(list(set(new_data.bbox_idx.cpu().numpy())))), len(roots))

            torch.cuda.synchronize() 
            st = time.time()
            new_bbox_idx = [0]
            count = 0
            for i in range(1, new_data.bbox_idx.size(0)):
                if new_data.bbox_idx[i] != new_data.bbox_idx[i - 1]:
                    count += 1
                new_bbox_idx.append(count)

            #print(len(new_bbox_idx), new_data.bbox_idx.size())
            
            #for i in range(len(new_bbox_idx)):
            #    print(new_bbox_idx[i], new_data.bbox_idx[i])
            new_data.bbox_idx = torch.tensor(np.array(new_bbox_idx), dtype=torch.long)
            #torch.cuda.synchronize() 
            #et = time.time()
            #print('get new bbox idx: ', et-st)
            
            return new_data
        
        #torch.cuda.synchronize() 
        #st = time.time()
        total_infer_time = 0
        new_data = build_data(data, slice_pos, slice_edge, slice_edge_super, slice_bbox)
        #torch.cuda.synchronize() 
        #et = time.time()
        #print('data building 0: ', et-st)
        torch.cuda.synchronize() 
        st = time.time()
        pred_cls, pred_bbox= self.forward(new_data, slices)
        torch.cuda.synchronize() 
        et = time.time()
        #print('inference 0: ', et-st)
        total_infer_time += et - st
        
        _, is_object = pred_cls.max(1)
        #print(is_object)
        has_object = (is_object == self.n_classes - 1) #(has_object == 1) & (is_object == self.n_classes - 1)

        #has_object[:] = True
        #print(has_object)
        slice_bbox_root = slice_bbox 

        slice_pos = []
        slice_edge = []
        slice_edge_super = []
        slice_bbox = []
        slice_image_bbox_child = [0]
        count = 0

        #torch.cuda.synchronize() 
        #st = time.time()
        for i in range(0, len(slice_root) - 1):
            start = slice_root[i]
            end = slice_root[i + 1]
            sub_roots = roots[start:end]
            for root in sub_roots:
                if not has_object[count]:
                    count += 1
                    continue
                for child in root.children:
                    slice_pos += list(range(child.value['idx_pos'][0] + slices['pos'][i], child.value['idx_pos'][1] + slices['pos'][i]))
                    slice_edge += list(range(child.value['idx_edge'][0] + slices['edge'][i], child.value['idx_edge'][1] + slices['edge'][i]))
                    slice_edge_super += list(range(child.value['idx_edge_super'][0] + slices['edge_super'][i], child.value['idx_edge_super'][1] + slices['edge_super'][i]))
                    slice_bbox.append(int(child.value['idx_bbox'] + slices['bbox'][i]))
                count += 1
            slice_image_bbox_child.append(len(slice_bbox))
        #torch.cuda.synchronize() 
        #et = time.time()
        #print('obtain slice 1: ', et-st)

        if len(slice_pos) == 0:
            pred_cls = pred_cls
            pred_bbox = pred_bbox
            slice_image_bbox = slice_image_bbox_root
            slice_bbox = slice_bbox_root
        else:
            #torch.cuda.synchronize() 
            #st = time.time()
            new_data = build_data(data, slice_pos, slice_edge, slice_edge_super, slice_bbox)
            #torch.cuda.synchronize() 
            #et = time.time()
            #print('data building 1: ', et-st)
            torch.cuda.synchronize() 
            st = time.time()
            pred_cls2, pred_bbox2 = self.forward(new_data, slices)
            torch.cuda.synchronize() 
            et = time.time()
            #print('inference 1: ', et-st)
            total_infer_time += et - st
            #print('total inference time', total_infer_time)
            #raise SystemExit
            
            def interleaf_pc(slice_p, slice_c, out_p, out_c):
                out = []
                s = [0]
                for i in range(len(slice_c) - 1):
                    start_root = slice_p[i]
                    end_root = slice_p[i + 1]
                    start_child = slice_c[i]
                    end_child = slice_c[i + 1]
                    out.append(out_p[start_root:end_root])
                    out.append(out_c[start_child:end_child])
                    s.append(s[-1] + end_root - start_root + end_child - start_child)
                return out, s
            
            pred_cls, slice_image_bbox = interleaf_pc(slice_image_bbox_root, slice_image_bbox_child, pred_cls, pred_cls2)
            pred_bbox, slice_image_bbox = interleaf_pc(slice_image_bbox_root, slice_image_bbox_child, pred_bbox, pred_bbox2)
            slice_bbox, slice_image_bbox = interleaf_pc(slice_image_bbox_root, slice_image_bbox_child, torch.tensor(slice_bbox_root), torch.tensor(slice_bbox))
            
            pred_cls = torch.cat(pred_cls, dim = 0)
            pred_bbox = torch.cat(pred_bbox, dim = 0)
            slice_bbox = torch.cat(slice_bbox, dim = 0)


        w = pred_bbox[:, 2] - pred_bbox[:, 0]
        h = pred_bbox[:, 3] - pred_bbox[:, 1]
        center_x = (pred_bbox[:, 2] + pred_bbox[:, 0]) / 2
        center_y = (pred_bbox[:, 3] + pred_bbox[:, 1]) / 2

        w = w * 1.05
        h = h * 1.05
        
        x0 = center_x - w / 2
        y0 = center_y - h / 2
        x1 = center_x + w / 2
        y1 = center_y + h / 2
        pred_bbox = torch.cat([x0.unsqueeze(1), y0.unsqueeze(1), x1.unsqueeze(1), y1.unsqueeze(1)], dim = 1)

        #torch.cuda.synchronize() 
        #print('predict overall:', time.time() - st_overal)
        
        return pred_cls, pred_bbox, None, slice_bbox, slice_image_bbox, None

class DetectionLoss(torch.nn.Module):
    def __init__(self, opt):
        super(DetectionLoss, self).__init__()

        if opt.classifier == 'softmax':
            self.cls_loss = torch.nn.CrossEntropyLoss()
        else:   
            self.cls_loss = torch.nn.BCELoss() 
        self.classifier = opt.classifier

    def forward(self, out, data):
        pred_cls = out[0]
        is_super = data.is_super
        gt_cls = data.labels.cuda()
        
        if self.classifier != 'softmax': 
            gt_cls = torch.zeros(pred_cls.size()).cuda().scatter_(1, gt_cls.unsqueeze(1), 1)

        l0 = self.cls_loss(pred_cls, gt_cls)
        loss = l0
        
        return {'loss': loss, 'loss_cls':l0}
