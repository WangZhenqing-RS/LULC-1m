### 1. Remote Sensing Image Inference

#### 1.1 Single Image Inference

Run inference on a single large-scale remote sensing image:

```bash
python infer_big_single.py
```

#### 1.2 Batch Image Inference

Run batch inference on multiple remote sensing images within a folder:

```bash
python infer_big_batch.py
```

#### 1.3 Clip Inference Results by Standard Map Sheets

Clip inference results using 1:10,000 standard map-sheet vector files:

```bash
python clip_raster_by_features.py
```

#### 1.4 Visualization with Color Table

Apply a color table to inference result rasters for visualization and cartographic display:

```bash
python gdal_color_table.py
```

---

### 2. Google Imagery Preprocessing

#### 2.1 Create City-Level Virtual Mosaic (VRT)

Organize downloaded Google imagery by city and generate virtual mosaic datasets:

```bash
python create_vrt_from_tif_folder.py
```

#### 2.2 Generate Buffered Map Sheets

Apply a 200 m outward buffer to 1:10,000 standard map-sheet vector files to avoid edge information loss during clipping:

```bash
python buffer.py
```

#### 2.3 Clip Google Imagery by Map Sheets

Clip virtual mosaic imagery using standard map-sheet vectors to produce map-compliant image tiles:

```bash
python clip_raster.py
```
