import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import argparse
import dataloader
import net

# import sys
# import time
# import torch.backends.cudnn as cudnn
# import numpy as np
# from torchvision import transforms

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--orig_images_path', type=str, default="Haze4K/train/gt/")
parser.add_argument('--hazy_images_path', type=str, default="Haze4K/train/haze/")
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshot_iter', type=int, default=200)
parser.add_argument('--cuda_index', type=str, default=2)
parser.add_argument('--snapshots_folder', type=str, default="snapshots2/")

config_para = parser.parse_args()
cuda_index = config_para.cuda_index

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    dehaze_net = net.DehazeNet().cuda(cuda_index)
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.DehazeLoader(config.orig_images_path, config.hazy_images_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda(cuda_index)
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()

    for epoch in range(config.num_epochs):
        for index, (img_orig, img_haze) in enumerate(train_loader):

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
