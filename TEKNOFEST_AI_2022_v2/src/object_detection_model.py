import warnings
warnings.filterwarnings("ignore")

import logging
import time

import requests

from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject

import torch
import numpy as np
import time
import cv2
import torch.nn as nn
from ensemble_boxes import *

from yolo.models.common import Conv
from yolo.utils.downloads import attempt_download
from yolo.utils.augmentations import letterbox
from yolo.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords)
from yolo.utils.plots import Annotator, colors
from yolo.utils.torch_utils import select_device, time_sync



class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        y = torch.cat(y, 1)  
        return y, None  

def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from yolo.models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  
        #if fuse:
        #    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  
        #else:
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval()) 

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace 
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set() 

    if len(model) == 1:
        return model[-1] 
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  


weights='./weights/zekai.pt'
device='0'
device = select_device(device)
model = attempt_load(weights, map_location=device)


class ObjectDetectionModel:
    def __init__(self, evaluation_server_url):
            print('Created Object Detection Model')
            logging.info('Created Object Detection Model')
            self.evaulation_server = evaluation_server_url

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1] 

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
        print(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')


    def process(self, prediction,evaluation_server_url, frame):
        self.download_image(evaluation_server_url + "media" + prediction.image_url, "./images/{}/".format(frame['video_name']))
       
        frame_results = self.detect(prediction, frame)
        
        return frame_results
    
    def yolo(self, image):
        imgsz=[1920]
        device='0'
        augment=False  
        half=False

        with torch.no_grad():
            imgsz *= 2 if len(imgsz) == 1 else 1
            classes = None

            device = select_device(device)
            half &= device.type != 'cpu'  

            stride, names = 64, [f'class{i}' for i in range(1000)] 
            
            stride = int(model.stride.max())  
            names = model.module.names if hasattr(model, 'module') else model.names  
            imgsz = check_img_size(imgsz, s=stride)
            if half:
                model.half()
            
            #model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters()))) 
            dt, seen = [0.0, 0.0, 0.0], 0

            t1 = time_sync()

            img = letterbox(image, imgsz, stride=stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)   
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  
            
            img /= 255  
            if len(img.shape) == 3:
                img = img[None] 
            t2 = time_sync()
            dt[0] += t2 - t1

            visualize = False
            pred = model(img, augment=augment, visualize=visualize)[0]
            
            t3 = time_sync()
            dt[1] += t3 - t2

            conf_thres = 0.25
            iou_thres = 0.45
            agnostic_nms =False
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)
            dt[2] += time_sync() - t3

            return img,pred,names, dt, seen
            
    def detect(self, prediction, frame):
        imgsz=[1920]

        image_name = prediction.image_url.split("/")[-1] 
        image = cv2.imread('./images/{}/{}'.format(frame['video_name'],image_name))
        yoloimage = image.copy()
        
        output_yolo = []
        img,pred,names, dt, seen = self.yolo(yoloimage)
        for i, det in enumerate(pred):  
            seen += 1
            im0= image.copy()

            annotator = Annotator(im0, line_width=3, example=str(names))
            
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    output_yolo.append((label, xyxy[0].cpu().detach().item(), xyxy[1].cpu().detach().item(), 
                                        xyxy[2].cpu().detach().item(), xyxy[3].cpu().detach().item(), conf.cpu().detach().item()))

        arac_count = 0 
        yaya_count = 0
        uap_count = 0
        uai_count = 0
        boxes_list = []
        labels_list = []
        scores_list = []
        for lab in output_yolo:
            if lab[0] == 'yaya':
                boxes_list.append((lab[1]/image.shape[1], lab[2]/image.shape[0], lab[3]/image.shape[1], lab[4]/image.shape[0]))
                scores_list.append((lab[5]))
                labels_list.append((1))
                yaya_count += 1
            elif lab[0] == 'arac':
                boxes_list.append((lab[1]/image.shape[1], lab[2]/image.shape[0], lab[3]/image.shape[1], lab[4]/image.shape[0]))
                scores_list.append((lab[5]))
                labels_list.append((0))
                arac_count += 1
            elif lab[0] == 'uai':
                boxes_list.append((lab[1]/image.shape[1], lab[2]/image.shape[0], lab[3]/image.shape[1], lab[4]/image.shape[0]))
                scores_list.append((lab[5]))
                labels_list.append((2))
                uai_count += 1
            elif lab[0] == 'uap':
                boxes_list.append((lab[1]/image.shape[1], lab[2]/image.shape[0], lab[3]/image.shape[1], lab[4]/image.shape[0]))
                scores_list.append((lab[5]))
                labels_list.append((3))
                uap_count += 1
            

        cv2.putText(im0, 'Arac: {}'.format(arac_count), (1600, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,125), 2, cv2.LINE_AA)
        cv2.putText(im0, 'Yaya: {}'.format(yaya_count), (1600, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(im0, 'UAI: {}'.format(uai_count), (1600, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(im0, 'UAP: {}'.format(uap_count), (1600, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        
        cv2.imwrite('./result/{}/{}'.format(frame['video_name'],image_name), im0)
        
        t = tuple(x / seen * 1E3 for x in dt) 
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


        arac_label, yaya_label, uap_label, uai_label = [], [], [], []
        arac_boxes, yaya_boxes, uap_boxes, uai_boxes = [], [], [], []
        for i in range(len(boxes_list)):
            if int(labels_list[i]) == 0:
                arac_boxes.append(boxes_list[i])
                arac_label.append(labels_list[i])
            elif int(labels_list[i]) == 1:
                yaya_boxes.append(boxes_list[i])
                yaya_label.append(labels_list[i])
            elif int(labels_list[i]) == 2:
                uai_boxes.append(boxes_list[i])
                uai_label.append(labels_list[i])
            elif int(labels_list[i]) == 3:
                uap_boxes.append(boxes_list[i])
                uap_label.append(labels_list[i])
        
        
        for i in range(len(arac_boxes)):
            x1 = int(arac_boxes[i][0]*image.shape[1])
            y1 = int(arac_boxes[i][1]*image.shape[0])
            x2 = int(arac_boxes[i][2]*image.shape[1])
            y2 = int(arac_boxes[i][3]*image.shape[0])
            landing_status = landing_statuses["Inis Alani Degil"]
            send_csl = classes["Tasit"],
            d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
            prediction.add_detected_object(d_obj)
            print(" Class =  Arac", " inis Durumu = ", "inis alani degil", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
        for i in range(len(yaya_boxes)):
            x1 = int(yaya_boxes[i][0]*image.shape[1])
            y1 = int(yaya_boxes[i][1]*image.shape[0])
            x2 = int(yaya_boxes[i][2]*image.shape[1])
            y2 = int(yaya_boxes[i][3]*image.shape[0])
            landing_status = landing_statuses["Inis Alani Degil"]
            send_csl = classes["Insan"],
            d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
            prediction.add_detected_object(d_obj)
            print(" Class =  Yaya", " inis Durumu = ", "inis alani degil", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
        for i in range(len(uai_boxes)):
            x1 = int(uai_boxes[i][0]*image.shape[1])
            y1 = int(uai_boxes[i][1]*image.shape[0])
            x2 = int(uai_boxes[i][2]*image.shape[1])
            y2 = int(uai_boxes[i][3]*image.shape[0])
            if len(yaya_boxes) > 0:
                for j in range(len(yaya_boxes)):
                    x = (((yaya_boxes[j][2] - yaya_boxes[j][0])/2)+yaya_boxes[j][0])*image.shape[1]
                    y = (((yaya_boxes[j][3] - yaya_boxes[j][1])/2)+yaya_boxes[j][1])*image.shape[0] 
                    if x1 <  x < x2 and y1 < y < y2:   
                        landing_status = landing_statuses["Inilemez"]
                        send_csl = classes["UAI"],
                        d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
                        prediction.add_detected_object(d_obj)
                        print(" Class =  UAI", " inis Durumu = ", "Inilmez", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
                    else:
                        landing_status = landing_statuses["Inilebilir"]
                        send_csl = classes["UAI"],
                        d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
                        prediction.add_detected_object(d_obj)
                        print(" Class =  UAI", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))           
            else:
                landing_status = landing_statuses["Inilebilir"]
                send_csl = classes["UAI"],
                d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
                prediction.add_detected_object(d_obj)
                print(" Class =  UAI", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
                
        for i in range(len(uap_boxes)):
            x1 = int(uap_boxes[i][0]*image.shape[1])
            y1 = int(uap_boxes[i][1]*image.shape[0])
            x2 = int(uap_boxes[i][2]*image.shape[1])
            y2 = int(uap_boxes[i][3]*image.shape[0])
            if len(yaya_boxes) > 0:
                for j in range(len(yaya_boxes)):
                    x = (((yaya_boxes[j][2] - yaya_boxes[j][0])/2)+yaya_boxes[j][0])*image.shape[1]
                    y = (((yaya_boxes[j][3] - yaya_boxes[j][1])/2)+yaya_boxes[j][1])*image.shape[0] 
                    if x1 <  x < x2 and y1 < y < y2:
                        landing_status = landing_statuses["Inilemez"]
                        send_csl = classes["UAP"],
                        d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
                        prediction.add_detected_object(d_obj)
                        print(" Class =  UAP", " inis Durumu = ", "Inilmez", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
                    else:
                        landing_status = landing_statuses["Inilebilir"]
                        send_csl = classes["UAP"],
                        d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
                        prediction.add_detected_object(d_obj)
                        print(" Class =  UAP", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
            else:
                landing_status = landing_statuses["Inilebilir"]
                send_csl = classes["UAP"],
                d_obj = DetectedObject(send_csl, landing_status, x1, y1, x2, y2)
                prediction.add_detected_object(d_obj)
                print(" Class =  UAP", " inis Durumu = ", "Inilebilir", " Coord = ({}, {})({}, {})".format(x1,y1,x2,y2))
        

        return prediction
