import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import math
import tqdm

'''
标准规范
http://c.gb688.cn/bzgk/gb/showGb?type=online&hcno=30799A9AF9BFA98E77D21ABC5F0984E2
'''
def generate_million_grid(min_lon, max_lon, min_lat, max_lat):
    """
    生成1:100万标准分幅网格
    :param min_lon: 区域最小经度
    :param max_lon: 区域最大经度
    :param min_lat: 区域最小纬度
    :param max_lat: 区域最大纬度
    :return: GeoDataFrame
    """
    # 1:100万图幅参数 (经差6°×纬差4°)
    lon_interval = 6.0
    lat_interval = 4.0
    
    features = []
    
    # 计算涉及的行列范围
    start_col = math.floor((min_lon - 180) / lon_interval)
    end_col = math.ceil((max_lon - 180) / lon_interval)
    start_row = math.floor(min_lat / lat_interval)
    end_row = math.ceil(max_lat / lat_interval)
    
    for col in tqdm.tqdm(range(start_col, end_col + 1)):
        for row in range(start_row, end_row + 1):
            # 计算当前网格四角坐标
            left = 180 + col * lon_interval
            right = left + lon_interval
            bottom = row * lat_interval
            top = bottom + lat_interval
            
            # 跳过区域外的网格
            if (right < min_lon or left > max_lon or 
                top < min_lat or bottom > max_lat):
                continue
            
            # 创建多边形
            polygon = Polygon([
                (left, bottom), (right, bottom),
                (right, top), (left, top), (left, bottom)
            ])
            
            # 生成标准图幅编号
            # 行号: A(0°-4°), B(4°-8°), ..., V(84°-88°)
            row_letter = chr(65 + int(row))  # A=65, B=66, etc.
            col_num = int(col) + 61
            grid_id = f"{row_letter}{col_num:02d}"
            
            features.append({
                'geometry': polygon,
                'grid_id': grid_id,
                'scale': '1M'
            })
    
    return gpd.GeoDataFrame(features, crs="EPSG:4326")

def generate_500k_grid(million_grid_gdf):
    """
    基于1:100万网格生成1:50万标准分幅网格
    :param grid_100k_gdf: 1:100万网格GeoDataFrame
    :return: GeoDataFrame
    """
    # 1:50万参数 (经差3°, 纬差2°)
    lon_interval = 3
    lat_interval = 2
    
    features = []
    
    for _, row_1m in tqdm.tqdm(million_grid_gdf.iterrows()):
        minx, miny, maxx, maxy = row_1m.geometry.bounds
        grid_1m_id = row_1m['grid_id']
        
        for row in range(2):  # 纬度方向2行
            for col in range(2):  # 经度方向2列
                left = minx + col * lon_interval
                right = left + lon_interval
                bottom = miny + row * lat_interval
                top = bottom + lat_interval
                
                polygon = Polygon([
                    [left, bottom], [right, bottom],
                    [right, top], [left, top], [left, bottom]
                ])
                
                # 生成1:1万标准编号: 1:100万编号 + B + 行号(3位) + 列号(3位)
                grid_id = f"{grid_1m_id}B{2-row:03d}{col+1:03d}"
                
                features.append({
                    'geometry': polygon,
                    'grid_id': grid_id,
                    'parent_id': grid_1m_id,
                    'scale': '100K'
                })
    
    return gpd.GeoDataFrame(features, crs="EPSG:4326")

def generate_100k_grid(million_grid_gdf):
    """
    基于1:100万网格生成1:10万标准分幅网格
    :param grid_100k_gdf: 1:100万网格GeoDataFrame
    :return: GeoDataFrame
    """
    # 1:10万参数 (经差30′=0.5°, 纬差20′≈0.333°)
    lon_interval = 0.5
    lat_interval = 1/3
    
    features = []
    
    for _, row_1m in tqdm.tqdm(million_grid_gdf.iterrows()):
        minx, miny, maxx, maxy = row_1m.geometry.bounds
        grid_1m_id = row_1m['grid_id']
        
        for row in range(12):  # 纬度方向12行
            for col in range(12):  # 经度方向12列
                left = minx + col * lon_interval
                right = left + lon_interval
                bottom = miny + row * lat_interval
                top = bottom + lat_interval
                
                polygon = Polygon([
                    [left, bottom], [right, bottom],
                    [right, top], [left, top], [left, bottom]
                ])
                
                # 生成1:1万标准编号: 1:100万编号 + D + 行号(3位) + 列号(3位)
                grid_id = f"{grid_1m_id}D{12-row:03d}{col+1:03d}"
                
                features.append({
                    'geometry': polygon,
                    'grid_id': grid_id,
                    'parent_id': grid_1m_id,
                    'scale': '100K'
                })
    
    return gpd.GeoDataFrame(features, crs="EPSG:4326")

def generate_50k_grid(million_grid_gdf):
    """
    基于1:100万网格生成1:5万标准分幅网格
    :param grid_100k_gdf: 1:100万网格GeoDataFrame
    :return: GeoDataFrame
    """
    # 1:5万参数 (经差15′（0.25°）, 纬差10′（≈0.1667°）)
    lon_interval = 0.25
    lat_interval = 1/6 
    
    features = []
    
    for _, row_1m in tqdm.tqdm(million_grid_gdf.iterrows()):
        minx, miny, maxx, maxy = row_1m.geometry.bounds
        grid_1m_id = row_1m['grid_id']
        
        for row in range(24):  # 纬度方向24行
            for col in range(24):  # 经度方向24列
                left = minx + col * lon_interval
                right = left + lon_interval
                bottom = miny + row * lat_interval
                top = bottom + lat_interval
                
                polygon = Polygon([
                    [left, bottom], [right, bottom],
                    [right, top], [left, top], [left, bottom]
                ])
                
                # 生成1:1万标准编号: 1:100万编号 + E + 行号(3位) + 列号(3位)
                grid_id = f"{grid_1m_id}E{24-row:03d}{col+1:03d}"
                
                features.append({
                    'geometry': polygon,
                    'grid_id': grid_id,
                    'parent_id': grid_1m_id,
                    'scale': '50K'
                })
    
    return gpd.GeoDataFrame(features, crs="EPSG:4326")

def generate_10k_grid(million_grid_gdf):
    """
    基于1:100万网格生成1:1万标准分幅网格
    :param grid_100k_gdf: 1:100万网格GeoDataFrame
    :return: GeoDataFrame
    """
    # 1:1万参数 (经差3'45"=0.0625°, 纬差2'30"≈0.04167°)
    lon_interval = 0.0625
    lat_interval = 1/24  # 2.5/60 ≈ 0.04167
    
    features = []
    
    for _, row_1m in tqdm.tqdm(million_grid_gdf.iterrows()):
        minx, miny, maxx, maxy = row_1m.geometry.bounds
        grid_1m_id = row_1m['grid_id']
        
        for row in range(96):  # 纬度方向96行
            for col in range(96):  # 经度方向96列
                left = minx + col * lon_interval
                right = left + lon_interval
                bottom = miny + row * lat_interval
                top = bottom + lat_interval
                
                polygon = Polygon([
                    [left, bottom], [right, bottom],
                    [right, top], [left, top], [left, bottom]
                ])
                
                # 生成1:1万标准编号: 1:100万编号 + G + 行号(3位) + 列号(3位)
                grid_id = f"{grid_1m_id}G{96-row:03d}{col+1:03d}"
                
                features.append({
                    'geometry': polygon,
                    'grid_id': grid_id,
                    'parent_id': grid_1m_id,
                    'scale': '10K'
                })
    
    return gpd.GeoDataFrame(features, crs="EPSG:4326")


# 主程序：生成中国范围内的标准分幅网格
if __name__ == "__main__":
    # 中国范围 (经度73°~135°, 纬度18°~54°)
    min_lon, max_lon = 73.0, 135.0
    min_lat, max_lat = 18.0, 54.0
    
    # 生成1:100万网格
    million_grid = generate_million_grid(min_lon, max_lon, min_lat, max_lat)
    million_grid.to_file("China_1M_Grid.shp", encoding='utf-8')
    
    # 生成1:10万网格
    grid_500k = generate_500k_grid(million_grid)
    grid_500k.to_file("China_500K_Grid.shp", encoding='utf-8')
    
    # 生成1:5万网格
    grid_10k = generate_50k_grid(million_grid)
    grid_10k.to_file("China_50K_Grid.shp", encoding='utf-8')

    # 生成1:1万网格
    grid_10k = generate_10k_grid(million_grid)
    grid_10k.to_file("China_10K_Grid.shp", encoding='utf-8')

    print("标准分幅网格生成完成！")