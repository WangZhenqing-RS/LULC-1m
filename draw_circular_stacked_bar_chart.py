import matplotlib.pyplot as plt
import numpy as np
import pandas

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# # 水体 1：
# # 耕地 2： 
# # 林地 3：
# # 草地 4：
# # 建筑 5：
# # 道路 6：
# # 裸土 7：
# # 其他 0：

# 示例数据
n = 33  # 城市数量
cities = ["Macao", "Hongkong", "Haikou", "Shanghai", "Nanjing", "Taiyuan", "Nanchang", "Guangzhou", "Zhengzhou", "Xining", 
          "Guiyang", "Wuhan", "Yinchuan", "Xi'an", "Jinan", "Hefei", "Changsha", "Fuzhou", "Tianjin", "Shenyang",
          "Lanzhou", "Urumqi", "Chengdu", "Shijiazhuang", "Beijing", "Hangzhou", "Hohhot", "Kunming", "Nanning", "Changchun",
          "Lhasa", "Harbin", "Chongqing"]

categories = ['water', 'cropland', 'forest', 'grassland', 'building', 'road', 'bare land'] # , 'others']
colors = ['#005ce6', '#ffff99', '#228b22', '#98fb98', '#cd0000', '#606060', '#cd853f'] # ,'#000000']

data_df = pandas.read_csv(r"H:\基础地理信息数据集\栅格数据集\谷歌地球影像\整理好数据-上传共享平台\proportion.csv")
print()

data = []
for city in cities:
    if city == "Xi'an":
        city = "Xian"
    data_city = data_df[data_df['name'] == city].to_numpy()[0,2:]
    print(city, data_city)
    data.append(data_city)
data = np.array(data, np.float32)
data = np.round(data,3)
print(data.shape)
print(data.dtype)
print(data)


# data = [[0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]] * 33
# data = np.array(data) / 2.
# print(data.shape)
# print(data.dtype)
# print(data)

angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

# 堆叠数据
bottom = np.zeros(n)
width = 2 * np.pi / n * 0.9

# 绘制
fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.set_yticklabels([]) # 隐藏径向刻度标签
ax.set_xticklabels([]) # 隐藏角度刻度标签
ax.grid(False)  # 关闭网格线，去掉八等分线

# 画堆叠柱
for i in range(len(categories)):
    ax.bar(angles, data[:, i], width=width, bottom=bottom, color=colors[i],
           label=categories[i], edgecolor='white')
    bottom += data[:, i]

# 在最外层添加城市标签
radius = 1.05
for angle, label in zip(angles, cities):
    # # 调整旋转角度，使其与柱子中心方向一致（考虑逆时针和偏移）
    # rotation = (np.pi/2 - angle) * 180 / np.pi
    rotation = 0
    ax.text(angle, radius, label, ha='center', va='center',
            rotation=rotation, rotation_mode='anchor')

# 图例
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
plt.tight_layout()
plt.savefig("LULC1m环状堆积图.png", dpi=300)
plt.show()