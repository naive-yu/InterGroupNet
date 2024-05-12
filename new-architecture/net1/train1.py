import torch
import torch.nn as nn
# import torchvision
import torch.optim
import os
import random
import argparse
import dataloader
import net1 as net
# import faulthandler
# # 在import之后直接添加以下启用代码即可
# faulthandler.enable()
# 后边正常写你的代码
# import sys
# import time
# import torch.backends.cudnn as cudnn
# import numpy as np
# from torchvision import transforms

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--orig_images_path', type=str, default="new-architecture/net1/Haze4K/train/gt/")
parser.add_argument('--hazy_images_path', type=str, default="new-architecture/net1/Haze4K/train/haze/")
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshot_iter', type=int, default=200)
parser.add_argument('--cuda_index', type=str, default=1)
parser.add_argument('--snapshots_folder', type=str, default="new-architecture/net1/snapshots1/")

config_para = parser.parse_args()
cuda_index = config_para.cuda_index
torch.cuda.empty_cache()


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('conv0') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('conv') != -1:
        m.qkv.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    dehaze_net = net.DehazeNet().cuda(cuda_index)
    dehaze_net.apply(weights_init)

    train_dataset1 = dataloader.DehazeLoader(config.orig_images_path, config.hazy_images_path, 0)
    train_dataset2 = dataloader.DehazeLoader(config.orig_images_path, config.hazy_images_path, 1)
    train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda(cuda_index)
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()
    train_list = []
    for (img_orig, img_haze) in train_loader1:
        train_list.append((img_orig, img_haze))
    # print(len(train_list))
    for (img_orig, img_haze) in train_loader2:
        train_list.append((img_orig, img_haze))
    for epoch in range(config.num_epochs):
        random.shuffle(train_list)
        # print(len(train_list))
        for index, (img_orig, img_haze) in enumerate(train_list):
            img_orig = img_orig.cuda(cuda_index)
            img_haze = img_haze.cuda(cuda_index)

            clean_image = dehaze_net(img_haze)

            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((index + 1) % config.display_iter) == 0:
                print("Loss at batch", index + 1, ":", loss.item())
        # 保存模型快照
        # record
        torch.save(
            dehaze_net.state_dict(),
            f"{config.snapshots_folder}DehazeNet_epoch{str(epoch)}.pth",
        )
        print(f"epoch{epoch} finished!")


if __name__ == "__main__":

    if not os.path.exists(config_para.snapshots_folder):
        os.mkdir(config_para.snapshots_folder)

    train(config_para)
