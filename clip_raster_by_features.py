import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os

def clip_raster_by_features(vector_path: str, raster_dir: str, output_dir: str):
    """
    使用面状矢量要素逐个裁剪栅格，保留覆盖率 > 0 的像素
    :param vector_path: 矢量文件路径（如 Shapefile）
    :param raster_path: 栅格文件路径（如 GeoTIFF）
    :param output_dir: 裁剪结果输出目录
    """
    # 读取矢量数据
    gdf = gpd.read_file(vector_path)

    # 遍历每个矢量要素
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        grid_id = row["grid_id"]
        if geometry.is_empty:  # 跳过空几何
            continue
        # 核心裁剪操作
        # try:
        raster_path = os.path.join(raster_dir, grid_id+".tif")
        # 读取栅格数据（注意：使用 with 语句确保资源正确释放）
        with rasterio.open(raster_path) as src:
            # 检查是否存在颜色表（通常在第1波段）
            if src.count == 1 and src.colormap(1):
                colortable = src.colormap(1)  # 获取颜色表字典
                colortable[0] = (255, 255, 255, 255) # 背景类设置为白色，便于出图
                has_colortable = True
            else:
                has_colortable = False
            # 检查矢量与栅格的坐标系一致性
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)  # 自动投影转换
            # 设置 all_touched=True 确保覆盖率 >0 的像素均被保留
            out_image, out_transform = mask(
                src,
                [geometry],
                crop=True,
                all_touched=True,
                nodata=src.nodata
            )
            # 构造输出元数据
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "driver": "GTiff",
                "compress": 'lzw',
            })
            # 输出文件名（按要素ID命名）
            output_path = os.path.join(output_dir, f"{grid_id}.tif")
            # 保存裁剪结果
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
                if has_colortable:
                    dest.write_colormap(1, colortable)  # 显式写入颜色表
            print(f"要素 {idx} 裁剪完成 → {output_path}")
        # except Exception as e:
        #     print(f"裁剪要素 {idx} 失败: {str(e)}")

# 调用示例
if __name__ == "__main__":

    vector_path = r"省会级城市1万比例尺标准图幅/北京.shp"
    raster_dir = r"10k标准图幅/北京LULC_1m_rgb"
    output_dir = r"10k标准图幅/Beijing"
    os.makedirs(output_dir, exist_ok=True)
    clip_raster_by_features(
        vector_path,
        raster_dir,
        output_dir
    )