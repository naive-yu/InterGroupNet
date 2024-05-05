import os
import cv2
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFilter
import glob
import random

random.seed(1143)


def generate_train_list(orig_images_path, hazy_images_path):
    # 训练集
    # print(f"{orig_images_path},{hazy_images_path}")
    train_list1 = []
    train_list2 = []
    tmp_dict = {}
    haze_image_list = sorted(glob.glob(hazy_images_path + "*.png"), key=lambda name: int(name.split('\\')[-1].split('/')[-1].split('.')[0].split('_')[0]))
    # print(haze_image_list)
    for image in haze_image_list:
        if os.name == 'posix':
            # print("当前程序在 Linux 系统上运行")
            image = image.split("/")[-1]  # 图片文件名
        elif os.name == 'nt':
            # print("当前程序在 Windows 系统上运行")
            image = image.split("\\")[-1]  # 图片文件名

        key = image.split("_")[0] + ".png"  # 该图片对应的原图文件名
        # print(key)
        # if key in tmp_dict.keys():
        tmp_dict[key] = image

    for idx, key in enumerate(tmp_dict.keys()):
        # print(key)
        if idx < 1500:
            train_list1.append([orig_images_path + key, hazy_images_path + tmp_dict[key]])
        else:
            train_list2.append([orig_images_path + key, hazy_images_path + tmp_dict[key]])
    random.shuffle(train_list1)
    random.shuffle(train_list2)
    return train_list1


class DehazeLoader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, mode='train'):
        self.data_list = generate_train_list(orig_images_path, hazy_images_path)

        if mode == 'train':
            # self.data_list = self.train_list
            print("Total train examples:", len(self.data_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        # print(f"{data_orig_path},{data_hazy_path}")
        # 通过路径获取图像
        data_orig = cv2.imread(data_orig_path)
        data_hazy = cv2.imread(data_hazy_path)
        
        # 图像维度 640*480*3
        # data_orig = cv2.resize(data_orig, (640, 480), interpolation=cv2.INTER_LANCZOS4)
        # data_hazy = cv2.resize(data_hazy, (640, 480), interpolation=cv2.INTER_LANCZOS4)

        data_orig = data_orig / 255.0
        data_hazy = data_hazy / 255.0

        # 转换为PyTorch Tensor，图像维度 3*640*480
        data_orig = torch.from_numpy(data_orig.transpose(2, 0, 1)).float()
        data_hazy = torch.from_numpy(data_hazy.transpose(2, 0, 1)).float()

        return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)
