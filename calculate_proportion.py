import numpy as np
import glob
import os
import tqdm
import pandas as pd
from osgeo import gdal

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

def count_pixel_values(image_folder):
    # 初始化统计数组
    total_counts = np.zeros(8, dtype=np.int64)  # 0-7共8个值
    
    # 遍历所有影像文件
    image_files = glob.glob(os.path.join(image_folder, "*.tif"))
    for file in tqdm.tqdm(image_files):
        # 加载影像
        img,_,_ = imread_gdal(file)
        # 统计当前影像的像素值
        counts = np.bincount(img.ravel(), minlength=8)
        # 累加到总统计
        total_counts += counts
    
    # 计算比例
    total_counts[0] = 0 # 0值基本全都是nodata
    total_pixels = total_counts.sum()
    proportions = total_counts / total_pixels
    
    return proportions, total_counts


dirs = glob.glob(r"整理好数据-上传共享平台/*")
city_names = []
proportions_s = []
for dir in dirs:
    city_name = dir.split("\\")[-1]
    city_names.append(city_name)
    print(city_name)
    proportions, counts = count_pixel_values(dir)
    proportions_s.append(proportions)
    for i in range(8):
        print(f"像素值 {i}: 数量={counts[i]}, 比例={proportions[i]:.4%}")

proportions_s = np.array(proportions_s)
data = {
    'name': city_names,
    'proportion_0': proportions_s[:,0],
    'proportion_1': proportions_s[:,1],
    'proportion_2': proportions_s[:,2],
    'proportion_3': proportions_s[:,3],
    'proportion_4': proportions_s[:,4],
    'proportion_5': proportions_s[:,5],
    'proportion_6': proportions_s[:,6],
    'proportion_7': proportions_s[:,7],
}
df = pd.DataFrame(data)
print(df)
# 保存CSV，不包含索引列
df.to_csv(r"整理好数据-上传共享平台/proportion.csv", index=False)