# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:04:28 2021

@author: DELL
"""

import torch.utils.data as D
from .dataset import OurDataset

def dataloader(image_paths,label_paths, mode, batch_size, shuffle=True, num_workers=8):
    dataset = OurDataset(image_paths, label_paths, mode)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader

