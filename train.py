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


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    dehaze_net = net.DehazeNet().cuda()
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.DehazeLoader(config.orig_images_path, config.hazy_images_path)
    val_dataset = dataloader.DehazeLoader(config.orig_images_path, config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers,
                                             pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()

    # 只写入一次验证集的原始图像和含雾图像
    # flag = True
    for epoch in range(config.num_epochs):
        for index, (img_orig, img_haze) in enumerate(train_loader):

            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)

            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((index + 1) % config.display_iter) == 0:
                print("Loss at batch", index + 1, ":", loss.item())
                # 保存模型快照
                # if ((index + 1) % config_para.snapshot_iter) == 0:
                #     torch.save(DehazeNet.state_dict(), config_para.snapshots_folder + "Epoch" + str(epoch) + '.pth')
            # record
            # Validation Stage
        # my_net.eval()
        # with torch.no_grad():
        #     for index, (img_orig, img_haze) in enumerate(val_loader):
        #         img_orig = img_orig.cuda()
        #         img_haze = img_haze.cuda()
        #         clean_image = my_net(img_haze)
        #
        #         if flag:
        #             torchvision.utils.save_image(img_orig, config.origin_output_folder + str(index + 1) + ".png")
        #             torchvision.utils.save_image(img_haze, config.haze_output_folder + str(index + 1) + ".png")
        #         torchvision.utils.save_image(clean_image, config.clean_output_folder + str(index + 1) + ".png")
        #     flag = False
        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "DehazeNet_epoch" + str(epoch) + ".pth")
        print(f"epoch{epoch} finished!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="data/images/")
    parser.add_argument('--hazy_images_path', type=str, default="data/data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    parser.add_argument('--origin_output_folder', type=str, default="original/")
    parser.add_argument('--clean_output_folder', type=str, default="clean_path/")
    parser.add_argument('--haze_output_folder', type=str, default="haze_image_path/")

    config_para = parser.parse_args()

    if not os.path.exists(config_para.snapshots_folder):
        os.mkdir(config_para.snapshots_folder)
    if not os.path.exists(config_para.sample_output_folder):
        os.mkdir(config_para.sample_output_folder)

    train(config_para)
