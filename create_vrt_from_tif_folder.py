import os
import glob
from osgeo import gdal

def create_vrt_from_tif_folder(input_folder, output_vrt_path):
    """
    为指定文件夹内的TIFF文件创建VRT文件
    
    参数:
        input_folder: 包含TIFF文件的文件夹路径
        output_vrt_path: 输出的VRT文件路径
    """
    # 获取文件夹内所有.tif文件
    tif_files = glob.glob(os.path.join(input_folder, '*.tif'))
    
    if not tif_files:
        print("警告: 没有找到任何TIFF文件")
        return
    
    # 创建VRT选项
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False)
    
    # 构建VRT
    vrt = gdal.BuildVRT(output_vrt_path, tif_files, options=vrt_options)
    
    # 必须关闭VRT以写入文件
    vrt = None
    
    print(f"成功创建VRT文件: {output_vrt_path}")

def create_vrt_from_tif_paths(tif_files, output_vrt_path):
    """
    为指定文件夹内的TIFF文件创建VRT文件
    
    参数:
        tif_files: 包含TIFF文件路径的list
        output_vrt_path: 输出的VRT文件路径
    """
    
    if not tif_files:
        print("警告: 没有找到任何TIFF文件")
        return
    
    # 创建VRT选项
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False)
    
    # 构建VRT
    vrt = gdal.BuildVRT(output_vrt_path, tif_files, options=vrt_options)
    
    # 必须关闭VRT以写入文件
    vrt = None
    
    print(f"成功创建VRT文件: {output_vrt_path}")

# 使用示例
if __name__ == "__main__":

    dirs = glob.glob("谷歌地球影像/*")
    for dir in dirs:
        dir_basename = os.path.basename(dir)
        tif_paths = glob.glob(dir + "/*/*/*nodata.tif")
        print("="*30)
        print(dir+".vrt")
        print(tif_paths)
        create_vrt_from_tif_paths(tif_paths, dir+".vrt")
        # break

