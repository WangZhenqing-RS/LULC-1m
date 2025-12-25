### 一、遥感影像LULC推理（Inference）

#### 1. 单景遥感影像推理

对单个大幅遥感影像进行模型推理：

```bash
python infer_big_single.py
```

#### 2. 批量遥感影像推理

对文件夹内的多景遥感影像进行批量推理：

```bash
python infer_big_batch.py
```

#### 3. 基于标准图幅的结果裁剪

使用 1:10000 标准图幅矢量文件对推理结果进行裁剪：

```bash
python clip_raster_by_features.py
```

#### 4. 推理结果可视化（颜色表）

为推理结果栅格添加颜色表，便于可视化与制图表达：

```bash
python gdal_color_table.py
```


### 二、谷歌影像预处理（Google Imagery Preprocessing）

#### 1. 构建城市级虚拟镶嵌影像（VRT）

将下载的谷歌影像按城市组织，并生成虚拟镶嵌影像：

```bash
python create_vrt_from_tif_folder.py
```

#### 2. 标准图幅缓冲区生成

对 1:10000 标准图幅矢量文件向外缓冲 200 米，用于避免裁剪边缘信息缺失：

```bash
python buffer.py
```

#### 3. 基于标准图幅裁剪谷歌影像

使用标准图幅矢量文件裁剪虚拟镶嵌影像，生成符合制图规范的影像数据：

```bash
python clip_raster.py
```
