import os
# import tensorflow as tf
import re
import glob
import cv2 
import numpy as np

class LabelFinder():
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

    def get_label(self, img_path):
        label_path = re.sub(self.img_dir, self.label_dir, img_path)
        label_path = re.sub(".jpg", ".txt", label_path)

        return label_path if os.path.exists(label_path) else None


def xywh2xyxy(bbox):
    "bbox is a list contains x,y,w,h"
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    xmin = x - w/2
    ymin = y - h/2
    xmax = x + w/2
    ymax = y + h/2
    return [xmin, ymin, xmax, ymax]


def load_yolo_label_xywh(label):
    """yolo anno [label, x, y, w, h] in normalized form 

    Args:
        anno_path (path): path to the yolo anno file

    Returns:
        dict: return a dict of yolo anno 
    """
    cls_ids = []
    bboxes = []
    with open(label, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            data = [float(x) for x in data]
            cls_ids.append(int(data[0])) 
            bbox = data[1:]
            bboxes.append(bbox)
            
    return cls_ids, bboxes


img_dir = "/home/walter/git/edgetpu_dev_repo/data/images/train"
label_dir = "/home/walter/git/edgetpu_dev_repo/data/labels/train"


class DataReader(object):
    def __init__(self, img_dir, label_dir) -> None:
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs = glob.glob(f"{self.img_dir}/*.jpg")
        pass

    def __len__(self):
        return len(self.imgs)

    def data_generator(self):
        for i in range(len(self.imgs)):
            img, label = self.load_img_and_label(i)
            yield img, label
    
    
    
    def load_img(self, img_path):
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return img
    
    def find_label(self, img_path):
        label_path = re.sub(self.img_dir, self.label_dir, img_path)
        label_path = re.sub(".jpg", ".txt", label_path)
        return label_path if os.path.exists(label_path) else None
    
    def load_label(self, label_path):
        """
        load [xywh class] label
        """
        cls_ids = []
        bboxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                data = line.strip().split()
                data = [float(x) for x in data]
                cls_ids.append(int(data[0])) 
                bbox = data[1:]
                bboxes.append(bbox)
        return cls_ids, bboxes
    

    def load_img_and_label(self, idx):
        img_path = self.imgs[idx]
        img = self.load_img(img_path)
        label_path = self.find_label(img_path)
        label = self.load_label(label_path)
        return img, label


data_reader = DataReader(img_dir, label_dir)

print(data_reader.load_img_and_label(1))