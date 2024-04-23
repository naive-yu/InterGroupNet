import torch
import torchvision
import torch.optim
import net
import numpy as np
from PIL import Image
import glob
import argparse
import os
import cv2
import torchvision.transforms.functional as F

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 参数配置器
parser = argparse.ArgumentParser(description='Performance')
parser.add_argument('--dehaze_dir', default='Haze4K/test/dehaze')
parser.add_argument('--original_dir', default='Haze4K/test/gt')
parser.add_argument('--haze_dir', default='Haze4K/test/haze')
parser.add_argument('--sample_dir', default='samples/')
parser.add_argument('--snapshot_model', default='snapshots/record-DehazeNet_epoch30.pth')

config = parser.parse_args()


def dehazeImage(my_net, haze_image_path, dehaze_path):
    # 读取雾化图像
    data_haze = cv2.imread(haze_image_path)
    # 将图像归一化到 [0, 1]
    if data_haze is None:
        print(f"Error: Unable to load image {haze_image_path}")
    data_haze = data_haze.astype(np.float32) / 255.0
    # 调整图像的通道顺序，从 (height, width, channels) 变为 (channels, height, width)
    data_haze = np.transpose(data_haze, (2, 0, 1))
    # 将图像数据转换为PyTorch Tensor，并添加一个批次维度
    data_haze = torch.from_numpy(data_haze).float().cuda().unsqueeze(0)

    # 使用模型进行去雾
    dehaze_image = my_net(data_haze)

    # 保存去雾后的图像
    dehaze_image = dehaze_image.squeeze().cpu().detach().numpy()
    dehaze_image = np.transpose(dehaze_image, (1, 2, 0))  # 将通道维度放在最后
    dehaze_image = (dehaze_image * 255.0).astype(np.uint8)  # 将像素值转换为0-255范围内的整数

    # 构建保存路径
    dehaze_file_name = haze_image_path.split('\\')[-1].split('_')[0] + '.png'
    dehaze_file_path = os.path.join(dehaze_path, dehaze_file_name)
    # print(dehaze_file_path)
    # 保存图像
    # torchvision.utils.save_image(dehaze_image, dehaze_file_path)
    cv2.imwrite(dehaze_file_path, dehaze_image)


def dataAnalysis(haze_path, origin_path, dehaze_path):
    score_psnr = 0
    score_ssim = 0
    file_name_list = os.listdir(haze_path)
    for idx, file_name in enumerate(file_name_list):
        # 获取图像路径
        haze_image_path = os.path.join(haze_path, file_name)
        origin_image_path = os.path.join(origin_path, file_name.split("_")[0] + '.png')
        dehaze_image_path = os.path.join(dehaze_path, file_name.split("_")[0] + '.png')

        # 读取原始图像和去雾后的图像
        # print(f"{origin_image_path}")
        origin_image = cv2.imread(origin_image_path).astype(np.float32) / 255.0
        dehaze_image = cv2.imread(dehaze_image_path).astype(np.float32) / 255.0

        # # 调整图像大小
        # (h, w, c) = origin_image.shape
        # dehaze_image = cv2.resize(dehaze_image, (w, h))  # 原图大小

        # 调整图像大小
        (h, w, c) = dehaze_image.shape
        origin_image = cv2.resize(origin_image, (w, h))  # 参考大小

        # 计算评估指标
        score_psnr += psnr(origin_image, dehaze_image)
        score_ssim += ssim(origin_image, dehaze_image, multichannel=True, channel_axis=-1, data_range=1.0)

        # 保存图像
        haze_image = cv2.imread(dehaze_image_path).astype(np.float32) / 255.0
        if (idx + 1) % 10 == 0:
            im1 = F.to_tensor(origin_image).unsqueeze(0)
            im2 = F.to_tensor(dehaze_image).unsqueeze(0)
            im3 = F.to_tensor(haze_image).unsqueeze(0)
            combined_image = torch.cat((im1, im2, im3), dim=0)
            torchvision.utils.save_image(combined_image, config.sample_dir + str(idx + 1) + ".jpg")

    avg_score_psnr = score_psnr / len(file_name_list)
    avg_score_ssim = score_ssim / len(file_name_list)
    return avg_score_psnr, avg_score_ssim


if __name__ == '__main__':
    # 读取参数
    dehaze_dir = config.dehaze_dir
    original_dir = config.original_dir
    haze_dir = config.haze_dir
    snapshot_model = config.snapshot_model

    # 导入快照模型
    dehaze_net = net.DehazeNet().cuda()
    dehaze_net.load_state_dict(torch.load(snapshot_model))

    # 测试模式
    dehaze_net.eval()
    print("test start....")
    with torch.no_grad():
        # print(glob.glob(f"{haze_dir}/*"))
        for image in glob.glob(f"{haze_dir}/*"):
            dehazeImage(dehaze_net, image, dehaze_dir)
    print("test end....")

    # 分析结果
    avg_psnr, avg_ssim = dataAnalysis(haze_dir, original_dir, dehaze_dir)
    print("===> Avg_PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg_SSIM: {:.4f} ".format(avg_ssim))
