import datetime
import torch
import torchvision
import torch.optim
import net2 as net1
import net2 as net2
import record as net3
import net4 as net4
import numpy as np
import pandas as pd
from PIL import Image
import glob
import argparse
import os
import cv2
import torchvision.transforms.functional as F
import torch.nn.functional as F2

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 参数配置器
dehaze_for_net_index = 2
parser = argparse.ArgumentParser(description='Performance')
# 拟合程度验证
# parser.add_argument('--dehaze_dir', default='Haze4K/train/dehaze')
# parser.add_argument('--original_dir', default='Haze4K/train/gt')
# parser.add_argument('--haze_dir', default='Haze4K/train/haze')
# 普通测试与泛化测试
# parser.add_argument('--dehaze_dir', default='Haze4K/test/dehaze')
# parser.add_argument('--original_dir', default='Haze4K/test/gt')
# parser.add_argument('--haze_dir', default='Haze4K/test/haze')
parser.add_argument('--dehaze_dir', default='data/dehaze')
parser.add_argument('--original_dir', default='data/images')
parser.add_argument('--haze_dir', default='data/data')
parser.add_argument('--sample_dir', default=f'samples{dehaze_for_net_index}/')
parser.add_argument('--result_file', default=f'result{dehaze_for_net_index}.csv')
# parser.add_argument('--snapshot_model_dir_or_file', default=f'snapshots{dehaze_for_net_index}/')
# parser.add_argument('--snapshot_model_dir_or_file', default=f'snapshots{dehaze_for_net_index}/DehazeNet_epoch199.pth')
# parser.add_argument('--snapshot_model_dir_or_file', default='snapshots2/DehazeNet_epoch137.pth')
parser.add_argument('--snapshot_model_dir_or_file', default='record-snapshots/DehazeNet_epoch198.pth')
parser.add_argument('--cuda_index', default=0)

config = parser.parse_args()
test_length = 3000

# num_gpus = torch.cuda.device_count()
# print(num_gpus)
cuda_index = config.cuda_index


def dehazeImage(my_net, haze_image_path, dehaze_path):
    # 读取雾化图像
    data_haze = cv2.imread(haze_image_path)

    # 将图像归一化到 [0, 1]
    if data_haze is None:
        print(f"[{datetime.datetime.now()}] Error: Unable to load image {haze_image_path}")
    data_haze = data_haze.astype(np.float32) / 255.0
    # data_haze = cv2.resize(data_haze, (640, 480))
    # 调整图像的通道顺序，从 (height, width, channels) 变为 (channels, height, width)
    data_haze = np.transpose(data_haze, (2, 0, 1))
    # 将图像数据转换为PyTorch Tensor，并添加一个批次维度
    data_haze = torch.from_numpy(data_haze).float().cuda(cuda_index).unsqueeze(0)

    # 使用模型进行去雾
    dehaze_image = my_net(data_haze)

    # 保存去雾后的图像
    dehaze_image = dehaze_image.squeeze().cpu().detach().numpy()
    dehaze_image = np.transpose(dehaze_image, (1, 2, 0))  # 将通道维度放在最后
    dehaze_image = (dehaze_image * 255.0).astype(np.uint8)  # 将像素值转换为0-255范围内的整数

    # 构建保存路径
    dehaze_file_name = ''
    if len(haze_image_path.split('_')) == 1:  # 对去雾图再去雾，图像路径改变
        dehaze_file_name = haze_image_path.split('\\')[-1]
    elif os.name == 'posix':
        # print("当前程序在 Linux 系统上运行")
        dehaze_file_name = haze_image_path.split('/')[-1].split('_')[0] + '.png'
    elif os.name == 'nt':
        # print("当前程序在 Windows 系统上运行")
        dehaze_file_name = haze_image_path.split('\\')[-1].split('_')[0] + '.png'

    # print(f'w {dehaze_file_name}')
    dehaze_file_path = os.path.join(dehaze_path, dehaze_file_name)
    # print(dehaze_file_path)
    # 保存图像
    # torchvision.utils.save_image(dehaze_image, dehaze_file_path)
    cv2.imwrite(dehaze_file_path, dehaze_image)


def dataAnalysis(haze_dir, original_dir, dehaze_dir):
    score_psnr = 0
    score_ssim = 0  # 针对400*400
    file_name_list = sorted(os.listdir(haze_dir)[:-1], key=lambda name: int(name.split('.')[0].split('_')[0]))[:test_length]
    # print(file_name_list)
    for idx, file_name in enumerate(file_name_list):
        # 获取图像路径
        haze_image_path = os.path.join(haze_dir, file_name)
        if len(file_name.split("_")) == 1:
            origin_image_path = os.path.join(original_dir, file_name)
            dehaze_image_path = os.path.join(dehaze_dir, file_name)
        else:
            origin_image_path = os.path.join(original_dir, file_name.split("_")[0] + '.png')
            dehaze_image_path = os.path.join(dehaze_dir, file_name.split("_")[0] + '.png')

        # 读取原始图像和去雾后的图像
        # print(f"{origin_image_path}")
        origin_image = cv2.imread(origin_image_path).astype(np.float32) / 255.0
        dehaze_image = cv2.imread(dehaze_image_path).astype(np.float32) / 255.0

        # # 调整图像大小
        # (h, w, c) = origin_image.shape
        # dehaze_image = cv2.resize(dehaze_image, (w, h))  # 原图大小

        # # 调整图像大小
        # (h, w) = origin_image.shape[:2]  # 获取原图的高度和宽度
        # dehaze_image = cv2.resize(dehaze_image, (w, h), interpolation=cv2.INTER_LANCZOS4)  # 调整为原图大小

        # 计算评估指标
        score_psnr += psnr(origin_image, dehaze_image)
        score_ssim += ssim(origin_image, dehaze_image, multichannel=True, channel_axis=-1, data_range=1.0)

        # 保存图像
        haze_image = cv2.imread(haze_image_path).astype(np.float32) / 255.0

        # # 仅用于循环去雾测试
        # if haze_image.shape[:2] != (h, w):
        #     haze_image = cv2.resize(haze_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
        # 仅用于循环去雾测试

        # print(f'{origin_image_path} {origin_image.shape} {dehaze_image.shape} {haze_image.shape}')
        if (idx + 1) % 10 == 0:
            im1 = F.to_tensor(origin_image).unsqueeze(0)
            im2 = F.to_tensor(dehaze_image).unsqueeze(0)
            im3 = F.to_tensor(haze_image).unsqueeze(0)
            combined_image = torch.cat((im1, im2, im3), dim=0)
            # torch.save(combined_image, 'tensor.pth')
            torchvision.utils.save_image(combined_image, config.sample_dir + str(idx + 1) + ".jpg")

    avg_score_psnr = score_psnr / len(file_name_list)
    avg_score_ssim = score_ssim / len(file_name_list)
    return avg_score_psnr, avg_score_ssim


def runTest(net, snapshot_model):
    # 测试模式
    net.load_state_dict(torch.load(snapshot_model, map_location=f'cuda:{cuda_index}'))
    net.eval()
    print(f"[{datetime.datetime.now()}] test start with {snapshot_model}")
    with torch.no_grad():
        # print(glob.glob(f"{haze_dir}/*"))
        for idx, image in enumerate(sorted(glob.glob(f"{haze_dir}/*"), key=lambda name: int(name.split('\\')[-1].split('/')[-1].split('.')[0].split('_')[0]))):
            if idx < test_length:
                dehazeImage(net, image, dehaze_dir)
            # # 仅测试使用
            # break
    print(f"[{datetime.datetime.now()}] test end with {snapshot_model}")


def analysis_out(file_path, DF):
    # pd.DataFrame(result).to_excel(util.stock_files_dataAnalysis_path, sheet_name='sheet1', index=False)
    try:
        if not os.path.exists(file_path):
            DF.to_csv(file_path, index=False)
        else:
            DF.to_csv(file_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Warning!")
        raise e


if __name__ == '__main__':
    # 读取参数
    snapshot_model_dir_or_file = config.snapshot_model_dir_or_file
    dehaze_dir = config.dehaze_dir
    original_dir = config.original_dir
    haze_dir = config.haze_dir
    result_file = config.result_file

    # 确保目录存在
    if not os.path.exists(dehaze_dir):
        os.mkdir(dehaze_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    # 导入快照模型
    net_num = snapshot_model_dir_or_file.split('/')[-2][-1]
    if not net_num.isdigit():
        dehaze_net = net1.DehazeNet().cuda(cuda_index)
    else:
        net_num = int(net_num)
        if net_num == 1:
            dehaze_net = net1.DehazeNet().cuda(cuda_index)
        elif net_num == 2:
            dehaze_net = net2.DehazeNet().cuda(cuda_index)
        elif net_num == 3:
            dehaze_net = net3.DehazeNet().cuda(cuda_index)
        else:
            dehaze_net = net4.DehazeNet().cuda(cuda_index)
        # 判断路径是目录还是文件
    try:
        if os.path.isfile(snapshot_model_dir_or_file):
            # 单文件不使用表格记录结果
            # runTest(dehaze_net, snapshot_model=snapshot_model_dir_or_file)
            # 分析结果
            avg_psnr, avg_ssim = dataAnalysis(haze_dir, original_dir, dehaze_dir)
            print(f"[{datetime.datetime.now()}] Avg_PSNR: {avg_psnr} dB, Avg_SSIM: {avg_ssim}")
        elif os.path.isdir(snapshot_model_dir_or_file):
            exist_model = []
            if os.path.exists(result_file):
                exist_model = list(pd.read_csv(result_file)['model'])
            # print(exist_model)
            for snapshot_model in sorted(os.listdir(snapshot_model_dir_or_file), key=lambda name: int(name.split('.')[0][15:])):
                # print(snapshot_model)
                if snapshot_model in exist_model:
                    continue
                # print(snapshot_model)
                df = pd.DataFrame(columns=['net', 'epoch', 'model', 'avg_psnr', 'avg_ssim'])
                snapshot_model_index = snapshot_model.split('.')[0].split('h')[-1]
                snapshot_model_file = os.path.join(snapshot_model_dir_or_file, snapshot_model)
                runTest(dehaze_net, snapshot_model=snapshot_model_file)
                # 分析结果并输出到csv文件
                avg_psnr, avg_ssim = dataAnalysis(haze_dir, original_dir, dehaze_dir)
                df.loc[len(df)] = {'net': f'net{net_num}', 'epoch': snapshot_model_index, 'model': snapshot_model, 'avg_psnr': avg_psnr,
                                   'avg_ssim': avg_ssim}
                print(f"[{datetime.datetime.now()}] Avg_PSNR: {avg_psnr} dB, Avg_SSIM: {avg_ssim}")
                # 测试专用
                # if idx >= 2:
                #     break
                analysis_out(result_file, df)
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Warning! net{net_num}")
        print(e)
        raise e
