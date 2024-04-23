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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 参数配置器
parser = argparse.ArgumentParser(description='Performance')
parser.add_argument('--clean_dir', default='clean')
parser.add_argument('--original_dir', default='original')
parser.add_argument('--haze_dir', default='haze')
parser.add_argument('--snapshot_model', default='snapshots/record-DehazeNet_epoch30.pth')
config = parser.parse_args()


def dehaze_image(my_net, haze_image_path, clean_path):

	data_haze = Image.open(haze_image_path)
	# data_haze = (np.asarray(data_haze) / 255.0)
	# 转换为 NumPy 数组，并归一化到 [0, 1]
	data_haze = np.array(data_haze, dtype=np.float32) / 255.0

	data_haze = torch.from_numpy(data_haze).float()
	data_haze = data_haze.permute(2, 0, 1)
	data_haze = data_haze.cuda().unsqueeze(0)

	clean_image = my_net(data_haze)
	# print(haze_image_path.split("/")[-1].split("_")[0])
	torchvision.utils.save_image(clean_image, clean_path + "/" + haze_image_path.split("\\")[-1])


def dataAnalysis(haze_path, clean_path):
	score_psnr = 0
	score_ssim = 0
	file_name_list = os.listdir(haze_path)
	for file_name in file_name_list:
		# print(im_path + '/' + file_name)
		im1 = cv2.imread(haze_path + '\\' + file_name)
		im2 = cv2.imread(clean_path + '\\' + file_name)

		(h, w, c) = im2.shape
		im1 = cv2.resize(im1, (w, h))  # reference size

		score_psnr += psnr(im1, im2)
		score_ssim += ssim(im1, im2, channel_axis=2, data_range=255,multichannel=True)

	avg_score_psnr = score_psnr / len(file_name_list)
	avg_score_ssim = score_ssim / len(file_name_list)
	return avg_score_psnr, avg_score_ssim


if __name__ == '__main__':
	clean_dir = config.clean_dir
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
		for image in glob.glob(f"{haze_dir}/*"):
			dehaze_image(dehaze_net, image, clean_dir)
	print("test end....")

	# 分析结果
	avg_psnr, avg_ssim = dataAnalysis(haze_dir, clean_dir)
	print("===> Avg_PSNR: {:.4f} dB ".format(avg_psnr))
	print("===> Avg_SSIM: {:.4f} ".format(avg_ssim))

