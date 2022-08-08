import argparse
from tarfile import BLOCKSIZE
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torch.nn as nn

import os
import csv
from ensemble_boxes import *

from yolo.utils.common import Conv, DWConv
from yolo.utils.datasets import LoadStreams, LoadImages, letterbox
from yolo.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolo.utils.plots import plot_one_box
from yolo.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolo.utils.google_utils import attempt_download


def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
  
    return iou



class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1) 
        return y, None  

def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # 
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set() 

    if len(model) == 1:
        return model[-1]  
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  

def yolov5(frame, save_img=False):
    with torch.no_grad():
        weights, view_img,  imgsz = 'yolo/weights/best.pt', 1, 1920
        
        set_logging()
        device = select_device('0')
        half = device.type != 'cpu'  

        model = attempt_load(weights, map_location=device)  
        imgsz = check_img_size(imgsz, s=model.stride.max()) 
        if half:
            model.half()  

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  
        _ = model(img.half() if half else img) if device.type != 'cpu' else None 

        img = letterbox(frame, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0 

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=True)[0]

        pred = non_max_suppression(pred, 0.7, 0.6, agnostic=True)
        t2 = time_synchronized()

    return img,pred,names,colors


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def retinanet(image, model_path):
    classes = {"arac":0, "uap":1, "uai":2}

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    
    image_orig = image

    rows, cols, cns = image.shape
    
    smallest_side = min(rows, cols)

    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    output = []
    with torch.no_grad():
        image = torch.from_numpy(image)
        
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        scores, classification, transformed_anchors = model(image.cuda().float())
        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)

            label_name = labels[int(classification[idxs[0][j]])]
            score = scores[j]
            output.append((label_name, x1, y1, x2, y2, score.cpu().detach().item()))
            
            caption = '{} {:.3f}'.format(label_name, score)
            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        return image_orig, output

if __name__=="__main__":

    image = cv2.imread("enes.jpg")
    retimage = image.copy()
    RGB = image.copy()
    
    start = time.time()

    output_yolo = []
    img,pred,names,colors = yolov5(RGB)
    for i, det in enumerate(pred):
        s, im0 =  '', RGB
        s += '%gx%g ' % img.shape[2:]  
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det): 
                print("yolov5 score = ", conf) 
                label = f'{names[int(cls)]}'
                plot_one_box(xyxy, im0, label=label, color=(255,0,0), line_thickness=1)
                output_yolo.append((label, xyxy[0].cpu().detach().item(), xyxy[1].cpu().detach().item(), 
                                    xyxy[2].cpu().detach().item(), xyxy[3].cpu().detach().item(), conf.cpu().detach().item()))
    
    
    retImg, output_ret = retinanet(retimage, 'retinanet/csv_retinanet_11.pt')
    

    boxes_list = []
    labels_list = []
    scores_list = []
    for label in output_ret:
        if label[0] == 'yaya':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((1))
        elif label[0] == 'arac':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((0))
        elif label[0] == 'uai':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((2))
        elif label[0] == 'uap':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((3))

    for label in output_yolo:
        if label[0] == 'yaya':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((1))
        elif label[0] == 'arac':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((0))
        elif label[0] == 'uai':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((2))
        elif label[0] == 'uap':
            boxes_list.append((label[1]/1920, label[2]/1080, label[3]/1920, label[4]/1080))
            scores_list.append((label[5]))
            labels_list.append((3))


    iou_thr = 0.67 
    skip_box_thr = 0.01
    sigma = 0.1
    weights = 1.0
    boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=[weights], iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    color = [(255,0,125), (255,0,0), (0,0,255), (0,255,0)]
    for i in range(len(boxes)):
       cv2.rectangle(image, (int(boxes[i][0]*1920), int(boxes[i][1]*1080)), (int(boxes[i][2]*1920), int(boxes[i][3]*1080)), (color[int(labels[i])]), 3)

    arac_count = 0 
    yaya_count = 0
    uap_count = 0
    uai_count = 0

    for lab in labels:
        if int(lab) == 0:
            arac_count += 1
        elif int(lab) == 1:
            yaya_count += 1
        elif int(lab) == 2:
            uai_count += 1
        elif int(lab) == 3:
            uap_count += 1

    cv2.putText(image, 'Arac: {}'.format(arac_count), (1650, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,125), 2, cv2.LINE_AA)
    cv2.putText(image, 'Yaya: {}'.format(yaya_count), (1650, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(image, 'UAI: {}'.format(uai_count), (1650, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'UAP: {}'.format(uap_count), (1650, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    

    cv2.imwrite('result/WBF_output.jpg', image)
    cv2.imwrite('result/yolo_output.jpg', im0)
    cv2.imwrite('result/retina_output.jpg', retImg)

    arac_label, yaya_label, uap_label, uai_label = [], [], [], []
    arac_boxes, yaya_boxes, uap_boxes, uai_boxes = [], [], [], []
    for i in range(len(boxes)):
        if int(labels[i]) == 0:
            arac_boxes.append(boxes[i])
            arac_label.append(labels[i])
        elif int(labels[i]) == 1:
            yaya_boxes.append(boxes[i])
            yaya_label.append(labels[i])
        elif int(labels[i]) == 2:
            uai_boxes.append(boxes[i])
            uai_label.append(labels[i])
        elif int(labels[i]) == 3:
            uap_boxes.append(boxes[i])
            uap_label.append(labels[i])
    
    classes = ["arac", "yaya", "uai", "uap"]
    for i in range(len(arac_boxes)):
        x1 = arac_boxes[i][0]*1920
        y1 = arac_boxes[i][1]*1080
        x2 = arac_boxes[i][2]*1920
        y2 = arac_boxes[i][3]*1080
        print(" Class =  Arac", " inis Durumu = ", "inis alani degil", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
    for i in range(len(yaya_boxes)):
        x1 = yaya_boxes[i][0]*1920
        y1 = yaya_boxes[i][1]*1080
        x2 = yaya_boxes[i][2]*1920
        y2 = yaya_boxes[i][3]*1080
        print(" Class =  Yaya", " inis Durumu = ", "inis alani degil", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
    for i in range(len(uai_boxes)):
        x1 = uai_boxes[i][0]*1920
        y1 = uai_boxes[i][1]*1080
        x2 = uai_boxes[i][2]*1920
        y2 = uai_boxes[i][3]*1080
        if len(yaya_boxes) > 0:
            for j in range(len(yaya_boxes)):
                x = (((yaya_boxes[j][2] - yaya_boxes[j][0])/2)+yaya_boxes[j][0])*1920 
                y = (((yaya_boxes[j][3] - yaya_boxes[j][1])/2)+yaya_boxes[j][1])*1080 
                if x1 <  x < x2 and y1 < y < y2:
                    print(" Class =  UAI", " inis Durumu = ", "inis alani degil", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
                else:
                    print(" Class =  UAI", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
        else:
            print(" Class =  UAI", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
    for i in range(len(uap_boxes)):
        x1 = uap_boxes[i][0]*1920
        y1 = uap_boxes[i][1]*1080
        x2 = uap_boxes[i][2]*1920
        y2 = uap_boxes[i][3]*1080
        if len(yaya_boxes) > 0:
            for j in range(len(yaya_boxes)):
                x = (((yaya_boxes[j][2] - yaya_boxes[j][0])/2)+yaya_boxes[j][0])*1920 
                y = (((yaya_boxes[j][3] - yaya_boxes[j][1])/2)+yaya_boxes[j][1])*1080 
                if x1 <  x < x2 and y1 < y < y2:
                    print(" Class =  UAP", " inis Durumu = ", "inis alani degil", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
                else:
                    print(" Class =  UAP", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
        else:
            print(" Class =  UAP", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))

    stop = time.time()
    print("Total time = ", stop - start)