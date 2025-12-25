import os
import glob
from osgeo import ogr

def buffer(input_shp, output_shp, buffer_distance=0.0001):
    # 读取SHP文件
    input_ds = ogr.Open(input_shp)
    input_lyr = input_ds.GetLayer()

    # 获取输入数据的SRS
    input_srs = input_lyr.GetSpatialRef()

    # 创建输出文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    output_ds = driver.CreateDataSource(output_shp)
    output_lyr = output_ds.CreateLayer(os.path.basename(output_shp).split(".")[0], srs=input_srs, geom_type=ogr.wkbPolygon)

    # 添加字段（复制输入图层的字段）
    input_feature_defn = input_lyr.GetLayerDefn()
    for i in range(input_feature_defn.GetFieldCount()):
        field_defn = input_feature_defn.GetFieldDefn(i)
        output_lyr.CreateField(field_defn)

    # 处理每个要素
    for feature in input_lyr:
        in_geom = feature.GetGeometryRef()
        buffer_geom = in_geom.Buffer(buffer_distance)  # 缓冲距离

        # 创建新要素
        out_feature = ogr.Feature(output_lyr.GetLayerDefn())
        out_feature.SetGeometry(buffer_geom)
        
        # 复制属性
        for i in range(input_feature_defn.GetFieldCount()):
            out_feature.SetField(input_feature_defn.GetFieldDefn(i).GetNameRef(),
                            feature.GetField(i))
        
        # 添加到输出图层
        output_lyr.CreateFeature(out_feature)
        out_feature = None

    # 清理
    input_ds = None
    output_ds = None

if __name__=="__main__":
    shp_paths = glob.glob("省会级城市1万比例尺标准图幅/*.shp")
    for shp_path in shp_paths:
        shp_name = os.path.basename(shp_path)
        output_shp = f"省会级城市1万比例尺标准图幅_buffer200m/{shp_name}"
        if not os.path.exists(output_shp):
            buffer(shp_path, output_shp, buffer_distance=0.002)