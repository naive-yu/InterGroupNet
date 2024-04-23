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
    train_list = []
    tmp_dict = {}
    haze_image_list = glob.glob(hazy_images_path + "*.png")

    for image in haze_image_list:
        image = image.split("\\")[-1]  # 图片文件名
        key = image.split("_")[0] + ".png"  # 该图片对应的原图文件名
        # print(key)
        if key in tmp_dict.keys():
            tmp_dict[key] = image

    for key in tmp_dict.keys():
        train_list.append([orig_images_path + key, hazy_images_path + tmp_dict[key]])

    random.shuffle(train_list)

    return train_list


class DehazeLoader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, mode='train'):
        self.train_list, self.val_list = generate_train_list(orig_images_path, hazy_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total train examples:", len(self.train_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        # 通过路径获取图像
        data_orig = cv2.imread(data_orig_path)
        data_hazy = cv2.imread(data_hazy_path)

        # 图像维度 640*480*3
        data_orig = cv2.resize(data_orig, (640, 480), interpolation=cv2.INTER_LANCZOS4)
        data_hazy = cv2.resize(data_hazy, (640, 480), interpolation=cv2.INTER_LANCZOS4)

        data_orig = data_orig / 255.0
        data_hazy = data_hazy / 255.0

        # 转换为PyTorch Tensor，图像维度 3*640*480
        data_orig = torch.from_numpy(data_orig.transpose(2, 0, 1)).float()
        data_hazy = torch.from_numpy(data_hazy.transpose(2, 0, 1)).float()

        return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)
