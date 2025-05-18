import sys
sys.path.append('yolov7')

import torch
from torch.serialization import add_safe_globals
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox
from yolov7.utils.torch_utils import select_device
import numpy as np
import cv2

# allow globals to overcome safe loading
from models.common import Concat
from models.common import Conv
from models.common import MP
from models.common import SP
from models.yolo import Detect
from models.yolo import Model
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import ModuleList
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample

safe_globals = [Concat, Conv, MP, SP,
Detect, Model, LeakyReLU, BatchNorm2d,
ModuleList, Sequential, Conv2d,
MaxPool2d, Upsample]

class YOLOv7Wrapper:
    def __init__(self, weights='yolov7-tiny.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, device=''):
        self.device = select_device(device)

        with torch.serialization.safe_globals(safe_globals):
            self.model = attempt_load(weights, map_location=self.device)

        self.model.eval()
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if self.device.type != 'cpu':
            self.model.half()  # to FP16

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def infer(self, img_bgr):
        img = letterbox(img_bgr, self.img_size, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.half() if self.device.type != 'cpu' else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0])[0]

        if pred is None:
            return []

        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img_bgr.shape).round()
        boxes = []
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
