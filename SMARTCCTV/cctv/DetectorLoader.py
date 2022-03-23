import time
import torch
import numpy as np
import torchvision.transforms as transforms
# from pytorchyolo.models import load_model
from queue import Queue
from threading import Thread

from cctv.Detection.Models import load_model # <- 추가
from cctv.Detection.Utils import non_max_suppression, ResizePadding
# from Detection.Utils import  ResizePadding


class TinyYOLOv3_onecls(object):
    """Load trained Tiny-YOLOv3 one class (person) detection model.
    Args:
        input_size: (int) Size of input image must be divisible by 32. Default: 416,
        config_file: (str) Path to Yolo model structure config file.,
        weight_file: (str) Path to trained weights file.,
        nms: (float) Non-Maximum Suppression overlap threshold.,
        conf_thres: (float) Minimum Confidence threshold of predicted bboxs to cut off.,
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 input_size=416,
                #  config_file='Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',
                #  weight_file='Models/yolo-tiny-onecls/best-model.pth',
                 config_file='cctv/yolov3-tiny.cfg', # 경로 바꿈
                 weight_file='cctv/yolov3-tiny.weights', # 경로 바꿈
                 nms=0.2,
                 conf_thres=0.45,
                 device='cuda'):
        self.input_size = input_size

        # self.model = Darknet(config_file).to(device)
        # self.model.load_state_dict(torch.load(weight_file))

        self.model=load_model(config_file, weight_file) #바꾼거
        # print("*"*30)#디버깅용
        # print("pass")
        self.model.eval()
        self.device = device

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)
        self.transf_fn = transforms.ToTensor()

    def detect(self, image, need_resize=True, expand_bb=5):
        """Feed forward to the model.
        Args:
            image: (numpy array) Single RGB image to detect.,
            need_resize: (bool) Resize to input_size before feed and will return bboxs
                with scale to image original size.,
            expand_bb: (int) Expand boundary of the boxs.
        Returns:
            (torch.float32) Of each detected object contain a
                [top, left, bottom, right, bbox_score, class_score, class]
            return `None` if no detected.
        """
        image_size = (self.input_size, self.input_size)
        if need_resize:
            image_size = image.shape[:2]
            image = self.resize_fn(image)

        image = self.transf_fn(image)[None, ...]
        scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]

        detected = self.model(image.to(self.device))
 

        detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]
        
  

        # 바뀐 것으로 score등 다 계산해서 나옴~~
        if detected is not None:
            detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2
            detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2
            detected[:, 0:4] /= scf # x,y 좌표

            detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
            detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)
            
        return detected


class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images)

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()







