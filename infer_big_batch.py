# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 22:55:38 2023

@author: wangzhenqing
"""

import os
import cv2
import sys
import math
import tqdm
import glob
import time
import torch
import datetime
import numpy as np
import torch.utils.data as D
import torch.nn.functional as F

from osgeo import gdal
from scipy import ndimage
from torchvision import transforms as T
# from skimage.morphology import binary_opening, binary_closing, disk
# from skimage.morphology import remove_small_holes, remove_small_objects

from nets.upernet import UPerNet

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# 读取宽、高、投影、仿射变换矩阵四个元信息
def get_meta(image_path):
    dataset = gdal.Open(image_path)
    if dataset == None:
        print(image_path + "文件无法打开.")
    # 栅格矩阵的列数
    width = dataset.RasterXSize 
    # 栅格矩阵的行数
    height = dataset.RasterYSize 
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    return width, height, proj, geotrans

# 读取影像
def imread_gdal(image_path, xoff=0, yoff=0, image_width=0, image_height=0):
    dataset = gdal.Open(image_path)
    if dataset == None:
        print(image_path + "文件无法打开.")
    # 栅格矩阵的列数
    width = dataset.RasterXSize 
    # 栅格矩阵的行数
    height = dataset.RasterYSize 
    if(image_width == 0 and image_height == 0):
        image_width = width
        image_height = height
    # 获取数据
    image = dataset.ReadAsArray(xoff, yoff, image_width, image_height)
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    return image, proj, geotrans

# 保存影像
def imwrite_gdal(save_path, image, image_geotrans=(0,0,0,0,0,0), image_proj=""):
    if 'int8' in image.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in image.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(image.shape) == 3:
        image_bands, image_height, image_width = image.shape
    elif len(image.shape) == 2:
        image = np.array([image])
        image_bands, image_height, image_width = image.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, int(image_width), int(image_height), int(image_bands), datatype, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    if(dataset!= None):
        dataset.SetGeoTransform(image_geotrans) # 写入仿射变换参数
        dataset.SetProjection(image_proj) # 写入投影
    for i in range(image_bands):
        dataset.GetRasterBand(i+1).WriteArray(image[i])
        # dataset.GetRasterBand(i+1).SetNoDataValue(0)  # 设置NoData值
    del dataset

def get_image_clip_offset(image_width, image_height, sliding_length, tile_length):
    image_clip_offset = []
    # x 对应 width , y 对应 height 
    # 高度上可整分图像块数目
    height_tile_number = int((image_height-sliding_length) / (tile_length-sliding_length))
    #  行上图像块数目
    width_tile_number = int((image_width - sliding_length) / (tile_length-sliding_length))
    for i in range(width_tile_number):
        for j in range(height_tile_number):
            # [x_off, y_off, result_x_off, result_y_off, result_x_end, result_y_end, clip_x_off, clip_y_off, tile_length]
            if i==0:
                if j==0:
                    image_clip_offset.append([0, 0, 
                                              0, 0, 
                                              tile_length-sliding_length//2, tile_length-sliding_length//2,
                                              0, 0,
                                              tile_length])
                else:
                    image_clip_offset.append([0, j*(tile_length-sliding_length), 
                                              0, j*(tile_length-sliding_length)+sliding_length//2, 
                                              tile_length-sliding_length//2, (j+1)*(tile_length-sliding_length)+sliding_length//2,
                                              0, sliding_length//2,
                                              tile_length])
            else:
                if j==0:
                    image_clip_offset.append([i*(tile_length-sliding_length), 0, 
                                              i*(tile_length-sliding_length)+sliding_length//2, 0,
                                              (i+1)*(tile_length-sliding_length)+sliding_length//2, tile_length-sliding_length//2,
                                              sliding_length//2, 0,
                                              tile_length])
                else:
                    image_clip_offset.append([i*(tile_length-sliding_length), j*(tile_length-sliding_length), 
                                              i*(tile_length-sliding_length)+sliding_length//2, j*(tile_length-sliding_length)+sliding_length//2,
                                              (i+1)*(tile_length-sliding_length)+sliding_length//2, (j+1)*(tile_length-sliding_length)+sliding_length//2,
                                              sliding_length//2, sliding_length//2,
                                              tile_length])
            
    # 行上的剩余像素数
    height_residue = (image_height-sliding_length) % (tile_length-sliding_length)
    # 列上的剩余像素数
    width_residue = (image_width-sliding_length) % (tile_length-sliding_length)
    # 向前裁剪最后一行
    if height_residue!=0:
        for i in range(width_tile_number):
            if i==0:
                image_clip_offset.append([0, image_height-tile_length, 
                                          0, image_height-tile_length+sliding_length//2, 
                                          tile_length-sliding_length//2, image_height, 
                                          0, sliding_length//2,
                                          tile_length])
            else:
                image_clip_offset.append([i*(tile_length-sliding_length), image_height-tile_length, 
                                          i*(tile_length-sliding_length)+sliding_length//2, image_height-tile_length+sliding_length//2,
                                          (i+1)*(tile_length-sliding_length)+sliding_length//2, image_height, 
                                          sliding_length//2, sliding_length//2,
                                          tile_length])
        # 向前裁剪最后一列
        if width_residue!=0:
            for j in range(height_tile_number):
                if j==0:
                    image_clip_offset.append([image_width-tile_length, 0, 
                                              image_width-tile_length+sliding_length//2, 0, 
                                              image_width, tile_length-sliding_length//2, 
                                              sliding_length//2, 0,
                                              tile_length])
                else:
                    image_clip_offset.append([image_width-tile_length, j*(tile_length-sliding_length), 
                                              image_width-tile_length+sliding_length//2, j*(tile_length-sliding_length)+sliding_length//2, 
                                              image_width, (j+1)*(tile_length-sliding_length)+sliding_length//2, 
                                              sliding_length//2, sliding_length//2,
                                              tile_length])
            # 向前裁剪右下角
            image_clip_offset.append([image_width-tile_length, image_height-tile_length, 
                                      image_width-tile_length+sliding_length//2, image_height-tile_length+sliding_length//2,
                                      image_width, image_height,
                                      sliding_length//2, sliding_length//2,
                                      tile_length])
        else:
            pass
    else:
        # 向前裁剪最后一列
        if width_residue!=0:
            for j in range(height_tile_number):
                if i==0:
                    image_clip_offset.append([image_width-tile_length, 0, 
                                              image_width-tile_length+sliding_length//2, 0, 
                                              image_width, tile_length-sliding_length//2, 
                                              sliding_length//2, sliding_length//2,
                                              tile_length])
                else:
                    image_clip_offset.append([image_width-tile_length, j*(tile_length-sliding_length), 
                                              image_width-tile_length+sliding_length//2, j*(tile_length-sliding_length)+sliding_length//2, 
                                              image_width, (j+1)*(tile_length-sliding_length)+sliding_length//2, 
                                              sliding_length//2, sliding_length//2,
                                              tile_length])
    return image_clip_offset, height_residue, width_residue



class InferBigDataset(D.Dataset):
    def __init__(self, image_path, image_clip_offset, image_scale=1):
        self.image_path = image_path
        self.image = imread_gdal(self.image_path)[0]
        self.image_clip_offset = image_clip_offset
        self.image_scale = image_scale
        self.len = len(image_clip_offset)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        x_off, y_off, result_x_off, result_y_off, result_x_end, result_y_end, clip_x_off, clip_y_off, tile_length = self.image_clip_offset[index]
        # image,_,_ = imread_gdal(self.image_path, x_off, y_off, tile_length, tile_length)
        image = self.image[:,y_off:y_off+tile_length, x_off:x_off+tile_length]
        image = np.transpose(image[:3],(1,2,0))
        if np.max(image)>255:
            per_max = np.nanpercentile(image, 99.8)
            image = image * 1.0 / per_max * 255
            image[image>255] = 255
            image[image<0] = 0
            image = image.astype(np.uint8)
        if self.image_scale != 1:
            zoom_scale = (self.image_scale, self.image_scale, 1)
            image = ndimage.interpolation.zoom(image, zoom=zoom_scale, order=1)
        return self.as_tensor(image), x_off, y_off, result_x_off, result_y_off, result_x_end, result_y_end, clip_x_off, clip_y_off
    
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_path, image_clip_offset, image_scale, batch_size, num_workers):
    dataset = InferBigDataset(image_path, image_clip_offset, image_scale)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader


def infer_big(model, save_path, test_loader, image_height, image_width, image_scale, proj, geotrans, tta=False):
    
    preds = np.zeros((image_height, image_width), np.uint8)
    for image, x_off, y_off, result_x_off, result_y_off, result_x_end, result_y_end, clip_x_off, clip_y_off in tqdm.tqdm(test_loader):
        image = image.to(DEVICE)
        with torch.no_grad():
            if tta:
                output1 = F.softmax(model(image)).cpu().numpy()
                output2 = F.softmax(model(torch.flip(image, [0, 3])))
                output2 = torch.flip(output2, [3, 0]).cpu().numpy()
                output3 = F.softmax(model(torch.flip(image, [0, 2])))
                output3 = torch.flip(output3, [2, 0]).cpu().numpy()
                output = (output1 + output2 + output3) / 3.
            else:
                output = F.softmax(model(image)).cpu().data.numpy()
        for i in range(output.shape[0]):
            
            pred = output[i]
            pred = pred.squeeze()
            pred = np.argmax(pred, axis=0)
            pred = pred.astype(np.uint8)
            if image_scale!=1:
                pred = cv2.resize(pred, (int(pred.shape[0]/image_scale), int(pred.shape[1]/image_scale)), cv2.INTER_NEAREST)
            
            # 将黑边赋予其他类
            image_i = image[i].cpu().numpy()
            image_sum = np.sum(image_i,0)
            pred[image_sum==0] = 0

            # pred[pred!=255] = 0
            # print(save_path)
            # cv2.imwrite(save_path, pred)
            preds[result_y_off[i]:result_y_end[i], result_x_off[i]:result_x_end[i],] = pred[\
                clip_y_off[i]:clip_y_off[i]+result_y_end[i]-result_y_off[i],
                clip_x_off[i]:clip_x_off[i]+result_x_end[i]-result_x_off[i]]
    imwrite_gdal(save_path, preds, geotrans, proj)
    


if __name__=="__main__":

    # 切片尺寸
    tile_length = 256
    # 滑动距离
    sliding_length = 128
    # 图像缩放尺度
    image_scale = 1
    # 批大小
    batch_size = 8
    # 进程数
    num_workers = 8
    # 使用测试时增强
    tta = True
    # 模型初始化
    backbone_name = "convnext_large"
    model = UPerNet(in_channels = 3, out_channels = 8, backbone_name = backbone_name, pretrained = False)
    model.to(DEVICE)
    # 训练好的模型地址
    model_path = f"upernet_convnext_large_lulc.pth"
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    image_dirs = glob.glob(r"H:\基础地理信息数据集\栅格数据集\谷歌地球影像\10k标准图幅\*")
    for image_dir in image_dirs:
        if "LULC" in image_dir:
            continue
        print(image_dir)
        big_image_paths = glob.glob(f"{image_dir}/*.tif")
        save_dir = f"{image_dir}LULC_1m"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        for big_image_path in tqdm.tqdm(big_image_paths):
            image_basename = os.path.basename(big_image_path)
            save_path = os.path.join(save_dir, image_basename)
            print(save_path)
            if os.path.exists(save_path):
                print("文件已存在！")
                continue
            image_width, image_height, proj, geotrans = get_meta(big_image_path)
            print(f"image_width: {image_width}, image_height: {image_height}")
            image_clip_offset, height_residue, width_residue = get_image_clip_offset(image_width, image_height, sliding_length, tile_length)
            test_loader = get_dataloader(big_image_path, image_clip_offset, image_scale, batch_size, num_workers)
            try:
                infer_big(model, save_path, test_loader, image_height, image_width, image_scale, proj, geotrans, tta)
            except Exception as e:
                print("推理失败！", e)       
