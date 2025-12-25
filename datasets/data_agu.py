# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import torchvision.transforms.functional as F

# 随机调节色调、饱和度值
def randomHueSaturationValue(image, hue_shift_limit=(-30, 30),
                             sat_shift_limit=(-5, 5),
                             val_shift_limit=(-15, 15), ratio=0.5):
    if np.random.random() < ratio:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        # hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        # hue_shift = np.uint8(hue_shift)
        # h_A += hue_shift
        # h_B += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.1, 0.1),
                           scale_limit=(-0.25, 0.25),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, ratio=0.5):
    if np.random.random() < ratio:
        image = np.flip(image, 1)
        mask = np.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, ratio=0.5):
    if np.random.random() < ratio:
        image = np.flip(image, 0)
        mask = np.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, ratio=0.5):
    if np.random.random() < ratio:
        image = np.rot90(image).copy()
        mask = np.rot90(mask).copy()

    return image, mask

def mutilScale(image, mask, scales=[0.5,0.75,1,1.25,1.5,1.75,2]):
    # 原始尺寸
    W, H = mask.shape
    # 随机选择缩放比例
    scale = random.choice(scales)
    new_W = int(W * scale)
    new_H = int(H * scale)
    
    # 缩放图像和掩码
    image = cv2.resize(image, (new_H, new_W))
    mask = cv2.resize(mask, (new_H, new_W), interpolation=cv2.INTER_NEAREST)
    
    # 处理尺寸不足的情况：填充至至少crop_size
    pad_w = max(W - new_W, 0)
    pad_h = max(H - new_H, 0)
    if pad_w > 0 or pad_h > 0:
        image = cv2.copyMakeBorder(image, pad_h//2, pad_h//2, pad_w//2, pad_w//2, cv2.BORDER_DEFAULT)
        mask = cv2.copyMakeBorder(mask, pad_h//2, pad_h//2, pad_w//2, pad_w//2, cv2.BORDER_DEFAULT)
        new_W, new_H = mask.shape
    
    # 随机裁剪
    top = random.randint(0, new_H - H)
    left = random.randint(0, new_W - W)
    image = image[top:H+top,left:W+left]
    mask = mask[top:H+top,left:W+left]
    return image.copy(), mask.copy()

def data_agu(image, label):
    
    image, label = randomHorizontalFlip(image, label)
    image, label = randomVerticleFlip(image, label)
    image, label = randomRotate90(image, label)
    image, label = mutilScale(image, label)
    # 拟合困难
    # image, label = randomShiftScaleRotate(image, label)
    # image = randomHueSaturationValue(image)
    return image.copy(), label.copy()


if __name__=="__main__":
    image_path = "17.png"
    label_path = r"label_17.png"
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = np.concatenate((image,image),2)
    label = cv2.imread(label_path,0)
    image, label = mutilScale(image, label,scales=[2])
    cv2.imwrite("image.png", image[:,:,3:6])
    cv2.imwrite("label.png", label)

