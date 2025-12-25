import os
import gc
import glob
import tqdm
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from rasterio.transform import from_origin
from osgeo import gdal, ogr
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")  # 启用UTF-8支持

def clip_raster(input_tif, shp_path, output_tif):
    """矢量裁剪栅格，显式释放内存"""
    # 强制释放未回收内存（可选）
    gc.collect()
    
    # 打开栅格（只读模式）
    input_ds = gdal.Open(input_tif, gdal.GA_ReadOnly)
    if not input_ds:
        raise RuntimeError("无法打开栅格文件")

    try:
        # 裁剪配置
        warp_options = gdal.WarpOptions(
            xRes=0.000005364418, # 输出分辨率（度）
            yRes=0.000005364418,
            resampleAlg=gdal.GRA_Cubic, # 三次卷积法
            cutlineDSName=shp_path,
            cropToCutline=True,
            dstNodata=0,  # 设置无效值为0
            creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"] # 压缩与大文件支持
        )
        
        # 执行裁剪并立即释放输出数据集
        output_ds = gdal.Warp(output_tif, input_ds, options=warp_options)
        if output_ds:
            output_ds.FlushCache()  # 强制写入磁盘
            output_ds = None  # 解除引用
    except:
        print(f"{input_tif}出现问题！")
    finally:
        # 确保输入数据集释放
        input_ds = None  # 关键！显式释放内存
    
    # 可选：再次触发垃圾回收
    gc.collect()

def clip_raster_rasterio(raster_path, vector_path, output_path):
    # 打开栅格数据
    with rasterio.open(raster_path) as src:
        # 读取矢量数据并统一投影
        gdf = gpd.read_file(vector_path).to_crs(src.crs)
        geometries = gdf.geometry.tolist()
        
        # 执行裁剪
        out_image, out_transform = mask(
            src, 
            geometries, 
            crop=True,        # 裁剪至矢量边界范围
            all_touched=True,  # 包含与边界接触的像元
            nodata=src.nodata   # 继承原始nodata值
        )
        
        # 更新元数据
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # 保存结果
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def feature2shp(shp_path, field_name, save_dir):
    # 打开矢量文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = driver.Open(shp_path, 0)  # 0表示只读
    layer = shp_ds.GetLayer()
    
    # 遍历每个要素
    for i in tqdm.tqdm(range(layer.GetFeatureCount())):
        feature = layer.GetFeature(i)
        field_value = feature.GetField(field_name)
        save_path = os.path.join(save_dir, f"{field_value}.shp")
        os.makedirs(save_dir, exist_ok=True)
        # 删除已存在文件
        if os.path.exists(save_path):
            driver.DeleteDataSource(save_path)
        # 创建新Shapefile
        out_ds = driver.CreateDataSource(save_path)
        out_layer = out_ds.CreateLayer("feature", geom_type=layer.GetGeomType())
        # 复制字段定义
        layer_defn = layer.GetLayerDefn()
        for j in range(layer_defn.GetFieldCount()):
            out_layer.CreateField(layer_defn.GetFieldDefn(j))
        # 复制几何和属性
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(feature.GetGeometryRef().Clone())
        for j in range(layer_defn.GetFieldCount()):
            out_feature.SetField(j, feature.GetField(j))
        out_layer.CreateFeature(out_feature)
        
        # 写入投影文件 (.prj)
        spatial_ref = layer.GetSpatialRef()
        with open(save_path.replace(".shp", ".prj"), "w") as prj_file:
            prj_file.write(spatial_ref.ExportToWkt())
        out_ds.FlushCache()  # 刷新缓存到磁盘
        out_ds = None


def mosaic_images_in_order(input_paths, output_path, resample_alg=gdal.GRA_Cubic, compress="LZW"):
    """
    按指定顺序镶嵌多个影像
    :param input_paths: 输入影像路径列表（顺序决定镶嵌优先级）
    :param output_path: 输出影像路径
    :param resample_alg: 重采样算法（默认最近邻）
    :param compress: 输出压缩格式（默认LZW）
    """
    # 1. 检查输入文件是否存在
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

    # 2. 验证投影一致性
    first_proj = gdal.Open(input_paths[0]).GetProjection()
    for path in input_paths[1:]:
        ds = gdal.Open(path)
        if ds.GetProjection() != first_proj:
            print(f"警告: {os.path.basename(path)} 的投影与第一幅影像不一致，强制转换到第一幅影像的投影")
        ds = None

    # 3. 执行镶嵌（按输入路径的顺序）
    options = gdal.WarpOptions(
        xRes=0.000005364418, # 输出分辨率（度）
        yRes=0.000005364418,
        format='GTiff',
        srcSRS=first_proj,
        dstSRS=first_proj,
        resampleAlg=resample_alg,
        srcNodata=0,  # 根据实际NoData值修改
        dstNodata=0,
        creationOptions=[f"COMPRESS={compress}", "TILED=YES", "BIGTIFF=IF_NEEDED"]
    )
    gdal.Warp(output_path, input_paths, options=options)

    # 4. 构建金字塔（提升加载效率）
    ds = gdal.Open(output_path, gdal.GA_Update)
    ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
    ds = None
    print(f"镶嵌完成！输出文件: {output_path}")


if __name__ == "__main__":

    input_tif = r"H:\基础地理信息数据集\栅格数据集\谷歌地球影像\海口.vrt"
    shp_path = r"H:\基础地理信息数据集\矢量数据集\行政区划数据集\2024版全国行政区划\省会级城市1万比例尺标准图幅_buffer200m\海口市.shp"
    feature_dir = r"H:\基础地理信息数据集\矢量数据集\行政区划数据集\2024版全国行政区划\省会级城市1万比例尺标准图幅_buffer200m\海口市_features"
    output_dir = r"H:\基础地理信息数据集\栅格数据集\谷歌地球影像\10k标准图幅\海口"
    feature2shp(shp_path, "grid_id", feature_dir)
    os.makedirs(output_dir, exist_ok=True)
    feature_paths = glob.glob(feature_dir+"/*.shp")
    for feature_path in tqdm.tqdm(feature_paths):
        output_tif = os.path.join(output_dir, os.path.basename(feature_path).replace(".shp",".tif"))
        # print(output_tif)
        clip_raster(input_tif, feature_path, output_tif)
