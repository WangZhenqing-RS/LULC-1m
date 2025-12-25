import os
import glob
import tqdm
from osgeo import gdal

def gdal_color_table(image_path, save_path):
    # 打开灰度图像
    src_ds = gdal.Open(image_path)
    gray_band = src_ds.GetRasterBand(1)
    gray_data = gray_band.ReadAsArray()

    # 创建输出图像（调色板格式）
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(save_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    out_ds.SetProjection(src_ds.GetProjection())
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())

    # 创建颜色表
    color_table = gdal.ColorTable()
    color_table.SetColorEntry(0, (0, 0, 0))  # 黑色 -> 其他
    color_table.SetColorEntry(1, (0, 92, 230))  # 深蓝色 -> 水体
    color_table.SetColorEntry(2, (255, 255, 153))  # 浅黄色 -> 耕地
    color_table.SetColorEntry(3, (34, 139, 34))  # 深绿色 -> 林地
    color_table.SetColorEntry(4, (152, 251, 152))  # 浅绿色 -> 草地
    color_table.SetColorEntry(5, (205, 0, 0))  # 暗红色 -> 建筑
    color_table.SetColorEntry(6, (96, 96, 96))  # 深灰色 -> 道路
    color_table.SetColorEntry(7, (205, 133, 63))  # 土棕色 -> 裸土

    # 应用颜色表到输出波段
    out_band = out_ds.GetRasterBand(1)
    out_band.SetRasterColorTable(color_table)
    out_band.WriteArray(gray_data)  # 写入原始灰度数据

    out_ds.FlushCache()  # 保存

if __name__=="__main__":
    pred_dir = r"H:\基础地理信息数据集\栅格数据集\谷歌地球影像\10k标准图幅\海口LULC_1m"
    pred_paths = glob.glob(pred_dir + "/*.tif")

    for pred_path in tqdm.tqdm(pred_paths):
        pred_dir = os.path.dirname(pred_path)
        output_dir = pred_dir.replace("LULC_1m","LULC_1m_rgb")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, os.path.basename(pred_path))
        print(save_path)
        if not os.path.exists(save_path):
            gdal_color_table(pred_path, save_path)
        # break
