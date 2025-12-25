# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import torch.utils.data as D

from .data_agu import data_agu
from torchvision import transforms as T
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

    
class OurDataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(label_paths)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        label = cv2.imread(self.label_paths[index], 0)
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        if self.mode == "train":
            # 普通数据增强
            image, label = data_agu(image, label)
            return self.as_tensor(image), label.astype(np.int64)
        
        elif self.mode == "val":
            return self.as_tensor(image), label.astype(np.int64)

    # 数据集数量
    def __len__(self):
        return self.len
    
    
    