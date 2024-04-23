import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# class DehazeNet(nn.Module):
#     def __init__(self):
#         super(DehazeNet, self).__init__()
#
#         # 第一个分组卷积层
#         self.conv1_1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, padding_mode='replicate')
#         self.relu1_1 = nn.ReLU(inplace=True)
#         self.conv1_2 = nn.Conv2d(32, 64, kernel_size=1)
#         self.relu1_2 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # 注意力模块
#         self.attention1 = AttentionModule(64, 32)
#         self.attention2 = AttentionModule(64, 16)
#
#         # 第二个卷积层和池化层
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # 上采样层和卷积层
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU(inplace=True)
#
#         self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         # 第一个分组卷积层和池化层
#         x = self.pool1(self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x)))))
#
#         # 注意力模块
#         x = self.attention1(x)
#         x = self.attention2(x)
#
#         # 第二个卷积层和池化层
#         x = self.pool2(self.relu2(self.conv2(x)))
#
#         # 上采样和卷积
#         x = self.upsample(x)
#         x = self.relu3(self.conv3(x))
#
#         # 最后的卷积层
#         x = self.conv4(x)
#
#         return x
#
#
# class AttentionModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(AttentionModule, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 使用全局平均池化获取注意力权重
#         weights = self.sigmoid(self.conv1(torch.mean(x, dim=(2, 3), keepdim=True)))
#
#         # 将注意力权重乘以输入特征图
#         x = x * weights
#
#         return x
# ===> Avg_PSNR: 18.6962 dB
# ===> Avg_SSIM: 0.8559  非边界填充

class DehazeNet(nn.Module):

    def __init__(self):
        super(DehazeNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 5, 1, 2, groups=3, bias=True, padding_mode='replicate')
        self.e_conv2 = nn.Conv2d(3, 3, 5, 1, 2, bias=True, padding_mode='replicate')
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True, padding_mode='replicate')
        self.e_conv4 = nn.Conv2d(6, 3, 5, 1, 2, bias=True, padding_mode='replicate')
        self.e_conv5 = nn.Conv2d(18, 3, 5, 1, 2, bias=True, padding_mode='replicate')
        self.attention = Attention2d(3, 3)
        self.spatial = SpatialOnlyBranch()

    def forward(self, x):
        source = [x]
        xatt = self.attention.forward(x)
        xsatt = self.spatial.forward(x)
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        # print(xatt.size())
        # print(x1.size())
        concat3 = torch.cat((x1, x2, x3, x4, xatt, xsatt), 1)
        x5 = self.relu(self.e_conv5(concat3))

        return self.relu((x5 * x) - x5 + 1)


class Attention2d(nn.Module):
    def __init__(self, in_planes, K, ):
        super(Attention2d, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, 1, )
        self.fc2 = nn.Conv2d(K, K, 1, )

    def forward(self, x):
        x1 = self.avgpool(x)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.softmax(x1)
        x = x * x1
        return x + x1


class SpatialOnlyBranch(nn.Module):

    def __init__(self, channel=3):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel, kernel_size=(1, 1))

        self.win_pool = nn.AvgPool2d(kernel_size=8, stride=4)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        )
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.local_attention = LocalityChannelAttention()

    def forward(self, x):
        b, c, h, w = x.size()
        # Spatial-only Self-Attention
        xlocal = self.local_attention(x)
        spatial_wv = self.sp_wv(x)  # bs,c,h,w
        spatial_wq = self.sp_wq(x)  # bs,c,h,w

        # spatial_wq=self.convchannel(spatial_wq)
        # spatial_wq = self.win_pool(spatial_wq)
        # spatial_wq = self.mlp(spatial_wq)
        #
        # spatial_wq = fun.relu(spatial_wq)
        # spatial_wq = fun.interpolate(spatial_wq, size=(h, w), mode='nearest')

        spatial_wq = self.agp(spatial_wq)  # bs,c,1,1
        spatial_wv = spatial_wv.reshape(b, c, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c)  # bs,1,c//2
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * xlocal
        spatial_out = self.sigmoid(spatial_out)
        spatial_out = spatial_out * x
        return spatial_out


class LocalityChannelAttention(nn.Module):
    def __init__(self, dim=3, winsize=8):
        super(LocalityChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        self.win_pool = nn.AvgPool2d(kernel_size=winsize, stride=winsize // 2)

    # self.gate = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.win_pool(x)
        y = self.mlp(y)
        # hard-sigmoid
        y = F.relu(y)
        y = F.interpolate(y, size=(h, w), mode='nearest')
        return y


# class DehazeNet(nn.Module):
#
#     def __init__(self):
#         super(DehazeNet, self).__init__()
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.e_conv1 = nn.Conv2d(3, 3, 5, 1, 2, bias=True)
#         self.e_conv2 = nn.Conv2d(3, 3, 5, 1, 2, bias=True)
#         self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
#         self.e_conv4 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
#         self.e_conv5 = nn.Conv2d(18, 3, 5, 1, 2, bias=True)
#
#         self.attention = attention2d(3, 3)
#         self.spatial = Spatial_only_branch()
#
#     def forward(self, x):
#         source = []
#         source.append(x)
#
#         xatt = self.attention.forward(x)
#         xsatt = self.spatial.forward(x)
#         x1 = self.relu(self.e_conv1(x))
#         x2 = self.relu(self.e_conv2(x1))
#
#         concat1 = torch.cat((x1, x2), 1)
#         x3 = self.relu(self.e_conv3(concat1))
#
#         concat2 = torch.cat((x2, x3), 1)
#         x4 = self.relu(self.e_conv4(concat2))
#
#         # print(xatt.size())
#         # print(x1.size())
#         concat3 = torch.cat((x1, x2, x3, x4, xatt, xsatt), 1)
#         x5 = self.relu(self.e_conv5(concat3))
#
#         clean_image = self.relu((x5 * x) - x5 + 1)
#
#         return clean_image
#
#
# class attention2d(nn.Module):
#     def __init__(self, in_planes, K, ):
#         super(attention2d, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.softmax = nn.Softmax(dim=1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_planes, K, 1, )
#         self.fc2 = nn.Conv2d(K, K, 1, )
#
#     def forward(self, x):
#         x1 = self.avgpool(x)
#         x1 = self.fc1(x1)
#         x1 = self.relu(x1)
#         x1 = self.fc2(x1)
#         x1 = self.softmax(x1)
#         x = x * x1
#         out = x + x1
#         return out
#
#
# class Spatial_only_branch(nn.Module):
#
#     def __init__(self, channel=3):
#         super().__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.sp_wv = nn.Conv2d(channel, channel, kernel_size=(1, 1))
#         self.sp_wq = nn.Conv2d(channel, channel, kernel_size=(1, 1))
#
#         self.win_pool = nn.AvgPool2d(kernel_size=8, stride=4)
#
#         self.mlp = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, channel, kernel_size=1, bias=False)
#         )
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.localattention = LocalityChannelAttention()
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # Spatial-only Self-Attention
#         xlocal = self.localattention(x)
#         spatial_wv = self.sp_wv(x)  # bs,c,h,w
#         spatial_wq = self.sp_wq(x)  # bs,c,h,w
#
#         # spatial_wq=self.convchannel(spatial_wq)
#         # spatial_wq = self.win_pool(spatial_wq)
#         # spatial_wq = self.mlp(spatial_wq)
#         #
#         # spatial_wq = F.relu(spatial_wq)
#         # spatial_wq = F.interpolate(spatial_wq, size=(h, w), mode='nearest')
#
#         spatial_wq = self.agp(spatial_wq)  # bs,c,1,1
#         spatial_wv = spatial_wv.reshape(b, c, -1)  # bs,c//2,h*w
#         spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c)  # bs,1,c//2
#         spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
#         spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
#         spatial_out = spatial_weight * xlocal
#         spatial_out = self.sigmoid(spatial_out)
#         spatial_out = spatial_out * x
#         return spatial_out
#
#
# class LocalityChannelAttention(nn.Module):
#     def __init__(self, dim=3, winsize=8):
#         super(LocalityChannelAttention, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False)
#         )
#         self.win_pool = nn.AvgPool2d(kernel_size=winsize, stride=winsize // 2)
#         # self.gate = nn.Sigmoid()
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         y = self.win_pool(x)
#         y = self.mlp(y)
#         # hard-sigmoid
#         y = F.relu(y)
#         y = F.interpolate(y, size=(h, w), mode='nearest')
#         return y
