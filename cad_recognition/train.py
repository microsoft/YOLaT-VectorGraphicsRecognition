import __init__
import torch
import numpy as np
import logging
import time
from itertools import repeat, product

#from torch_geometric.data import DataLoader, DataListLoader
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch_geometric.transforms as T
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.data import InMemoryDataset
from config import OptInit
from sklearn.metrics import confusion_matrix
import torchvision


from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
from utils import optim
from utils.det_util import get_batch_statistics, ap_per_class

from Datasets.graph_dict3 import SESYDFloorPlan as CADDataset
from architecture3cc_rpn_gp_iter2 import SparseCADGCN, DetectionLoss



def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def collate(data_list):
    r"""Collates a python list of data objects to the internal storage
    format of :class:`torch_geometric.data.InMemoryDataset`."""
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if isinstance(item[key], Tensor) and item[key].dim() > 0:
            cat_dim = item.__cat_dim__(key, item[key])
            cat_dim = 0 if cat_dim is None else cat_dim
            s = slices[key][-1] + item[key].size(cat_dim)
        elif isinstance(item[key], list):
            s = slices[key][-1] + len(item[key])
        else:
            s = slices[key][-1] + 1
        slices[key].append(s)
    #print(data['roots'], slices['roots'])
    
    if hasattr(data_list[0], '__num_nodes__'):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)
    
    for key in keys:
        item = data_list[0][key]
        if isinstance(item, Tensor) and len(data_list) > 1:
            if item.dim() > 0:
                cat_dim = data.__cat_dim__(key, item)
                cat_dim = 0 if cat_dim is None else cat_dim
                data[key] = torch.cat(data[key], dim=cat_dim)
            else:
                data[key] = torch.stack(data[key])
        elif isinstance(item, Tensor):  # Don't duplicate attributes...
            data[key] = data[key][0]
        elif isinstance(item, int) or isinstance(item, float):
            data[key] = torch.tensor(data[key])
        elif isinstance(item, list):
            new_list = []
            for item in data[key]:
                new_list += item
            data[key] = new_list
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices

def main():
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader ...')

    train_dataset = CADDataset(opt.data_dir, opt, partition = 'train', data_aug = opt.data_aug, do_mixup = opt.do_mixup, drop_edge = opt.drop_edge)
    train_loader = DataLoader(train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=8, 
        collate_fn = collate)
    
    test_dataset = CADDataset(opt.data_dir, opt, partition = 'test', data_aug = False, do_mixup = False, drop_edge = False)
    test_loader = DataLoader(test_dataset, 
        batch_size=opt.batch_size * 2, 
        shuffle=False, 
        num_workers=8, 
        collate_fn = collate)

#    if opt.multi_gpus:
#        train_loader = DataListLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
#    else:
#        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    opt.n_classes = len(list(set(train_dataset.class_dict.values())))
    opt.in_channels = test_dataset[0].x.shape[1]
    opt.n_objects = train_dataset.n_objects

    logging.info('===> Loading the network ...')
       
    model = SparseCADGCN(opt).to(opt.device)
    

    if opt.multi_gpus:
        model = DataParallel(SparseDeepGCN(opt)).to(opt.device)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    logging.info(model)

    logging.info('===> Init the optimizer ...')
    criterion = DetectionLoss(opt) #torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay = opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    logging.info('===> Init Metric ...')
    opt.losses = AverageMeter()
    # opt.test_metric = miou
    opt.test_values = AverageMeter()
    opt.test_value = 0.

    logging.info('===> start training ...')
    for _ in range(opt.total_epochs):
        opt.epoch += 1
        train(model, train_loader, optimizer, scheduler, criterion, opt)
        if opt.epoch % 1 == 0 and opt.epoch >= 0:
            test_value = test(model, test_loader, criterion, opt)
        scheduler.step()
    logging.info('Saving the final model.Finish!')


def train(model, train_loader, optimizer, scheduler, criterion, opt):
    model.train()
    for i, (data, slices) in enumerate(train_loader):
        opt.iter += 1

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
            elif 'bbox_idx' in key:
                bbox_offset = slices['labels']
                s = slices[key]
                #print(key, s)
                o = getattr(data, key)               
                for i_s in range(0, len(s) - 1):
                    start = s[i_s]
                    end = s[i_s + 1]
                    o[start:end] += bbox_offset[i_s]
                setattr(data, key, o)

        #raise SystemExit

        # ------------------ zero, output, loss
        optimizer.zero_grad()

        if not hasattr(data, 'edge_control'):
            data.edge_control = None
            #data.edge_pos = None


        out = model(data, slices)


        if opt.arch == 'centernet' or opt.arch == 'votenet':
            loss_dict = criterion(out[0], out[1], 
                data.labels.cuda(), 
                data.bbox.cuda(), 
                data.pos.cuda(),
                data.is_control.cuda())
        else:
            loss_dict = criterion(out, data)
        
        # ------------------ optimization
        loss_dict['loss'].backward()
        optimizer.step()

        opt.losses.update(loss_dict['loss'].item())
        # ------------------ show information
        if opt.iter % opt.print_freq == 0:
            output_string = 'Epoch:{}  Iter:{}[{}/{}]  LossMean:{Losses.avg: .4f} '.format(
                opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses)
            for key in loss_dict:
                output_string +='{}:{:.4f} '.format(key, loss_dict[key])
            output_string += 'lr:{:.4f}'.format(scheduler.get_last_lr()[0])
            logging.info(output_string)
            opt.losses.reset()

        # ------------------ tensor board log
        info = {
            'test_value': opt.test_value,
            'lr': scheduler.get_last_lr()[0]
        }
        for key in loss_dict:
            info[key] = loss_dict[key]
        for tag, value in info.items():
            opt.writer.add_scalar(tag, value, opt.iter)

    #raise SystemExit
    # ------------------ save checkpoints
    # min or max. based on the metrics
    is_best = (opt.test_value > opt.best_value)
    opt.best_value = max(opt.test_value, opt.best_value)

    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    # optim_cpu = {k: v.cpu() for k, v in optimizer.state_dict().items()}
    save_checkpoint({
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_value': opt.best_value,
    }, is_best, opt.ckpt_dir, opt.postname)


def test(model, test_loader, criterion, opt):
    opt.test_values.reset()
    model.eval()
    with torch.no_grad():
        sample_metrics = [[] for i in  range(10)]
        labels = []
        test_loss = {}

        n_true = 0
        n_total = 0

        y_true = []
        y_pred = []


        overall_time = 0
        for i_batch, (data, slices) in enumerate(test_loader):
            print(i_batch)
            torch.cuda.synchronize() 
            start_time = time.time()
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
            
            if not hasattr(data, 'edge_control'):
                data.edge_control = None
                #data.edge_pos = None

            
            out = model.predict(data, slices)
            
            data.labels = data.labels[out[3]]
            data.has_obj = data.has_obj[out[3]]
            slices['bbox'] = out[4]
            
            loss_dict = criterion(out, data)
            pred_cls = out[0]
            pred_coord = out[1]
            pred_coord_max = out[-1]
        
            
            _, pred_label = pred_cls.max(1)
            n_true += (pred_label == data.labels.cuda()).sum(0)
            n_total += pred_label.size(0)

            y_pred.append(pred_label.cpu().numpy())
            y_true.append(data.labels.cpu().numpy())

            if pred_coord_max is not None:
                pred_coord = pred_coord_max
            
            for key in loss_dict:
                if key not in test_loss:
                    test_loss[key] = []
                test_loss[key].append(loss_dict[key].cpu().data)

            #print(slices)
            image_slice = slices['x']
            label_slice = slices['gt_labels']

            new_pred_coord_list = []
            new_pred_cls_list = []

            for i in range(0, len(image_slice) - 1):
                start = image_slice[i]
                end = image_slice[i + 1]
                is_control_mask = ~data.is_control[start:end].squeeze()
                pos_img = data.pos[start:end][is_control_mask].cuda()

                t_start = slices['bbox'][i]
                t_end = slices['bbox'][i + 1]
                pred_coord_img = pred_coord[t_start:t_end]
                pred_cls_img = pred_cls[t_start:t_end]

                #print(start, end, pred_cls_img.size(), pred_cls.size(), 'fooo')

                #coord = data.bbox[start:end][is_control_mask]
                #center = (coord[:, 0:2] + coord[:, 2:]) / 2.0
                #wh = coord[:, 2:] - coord[:, 0:2]

                start = label_slice[i]
                end = label_slice[i + 1]
                gt_coord_img = data.gt_bbox[start:end]
                gt_coord_img[:, 0] *= data.width[i]
                gt_coord_img[:, 2] *= data.width[i]
                gt_coord_img[:, 1] *= data.height[i]
                gt_coord_img[:, 3] *= data.height[i]
                gt_cls_img = data.gt_labels[start:end].unsqueeze(1)
                targets = torch.cat((torch.zeros((gt_cls_img.size(0), 1)), 
                    gt_cls_img, gt_coord_img), dim = 1)
                labels += data.gt_labels[start:end].tolist()

                #print(pred_coord_img)
                pred_coord_img[:, 0] *= data.width[i]
                pred_coord_img[:, 2] *= data.width[i]
                pred_coord_img[:, 1] *= data.height[i]
                pred_coord_img[:, 3] *= data.height[i]

                #pred_cls_img = torch.cat((torch.ones(pred_cls_img.size(0), 1).cuda(), pred_cls_img), dim = 1)
                if opt.classifier == 'softmax':
                    pred_cls_img = F.softmax(pred_cls_img, dim = 1)

                pred_cls_img = torch.cat((1 - pred_cls_img[:, -1][:, None], pred_cls_img[:, 0:-1]), dim = 1)
                pred = torch.cat((pred_coord_img, pred_cls_img), dim = 1).unsqueeze(0)
                

                outputs = non_max_suppression(pred, conf_thres=0.0, iou_thres=0.5)
                outputs = [x.cpu() for x in outputs]

                

                iou_ths = np.linspace(0.5, 0.95, 10)
                for i_th, th in enumerate(iou_ths):
                    sample_metrics[i_th] += get_batch_statistics(outputs, targets, iou_threshold=th)

                
            #if i_batch == 0: break
        

        iou_ths = np.linspace(0.5, 0.95, opt.map_step)
        AP_total = 0
        output_str = ''
        for i in range(opt.map_step):
            if len(sample_metrics[i]) == 0:  # no detections over whole validation set.
                return None
            
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics[i]))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
            #print(AP)
            #test_value = test_metric(out.max(dim=1)[1], gt, opt.n_classes)
            #opt.test_values.update(test_value, opt.batch_size)
            #print(test_loss)
            output_str += 'Epoch: [{0}]\t Iter: [{1}]\t''MAP@{2:.2f}: {3:.4f}\t'.format(
                opt.epoch, opt.iter, iou_ths[i], np.mean(AP))
            output_str += 'Top1 Acc@{0:.2f}:{1:.4f}\t'.format(iou_ths[i], n_true * 1.0 / n_total)
            output_str += '\n'
            AP_total += np.mean(AP)

        overall_time /= len(test_loader.dataset)
        
        output_str += 'Epoch: [{0}]\t Iter: [{1}]\t''MAP@ALL: {2:.4f}\t inference_Time: {3:.4f}   '.format(
                opt.epoch, opt.iter,  AP_total / 10, overall_time * 1000)

        for key in test_loss:
            output_str += '{0}:{1:.4f}\t'.format(key, np.mean(test_loss[key]))
        logging.info(output_str + '\n')

        y_pred = np.concatenate(y_pred, axis = 0)
        y_true = np.concatenate(y_true, axis = 0)
        m = confusion_matrix(y_true, y_pred)

        cate_names = [''] * len(list(test_loader.dataset.class_dict.keys()))
        print()
        output_str = '          '
        for key in test_loader.dataset.class_dict:
            cate_names[test_loader.dataset.class_dict[key]] = key
        for cate in cate_names:
            output_str += '{:>10}'.format(cate)
        print(output_str)
        for i, row in enumerate(m):
            output_str = '{:>10}'.format(cate_names[i])
            for m in row:
                output_str += '{:10d}'.format(m)
            print(output_str)


    opt.test_value = np.mean(AP)
    return opt.test_value

if __name__ == '__main__':
    main()


