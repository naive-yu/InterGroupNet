import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

torch.set_printoptions(profile="full")
cuda_index = 3


class DehazeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 创建指数级增长的卷积核注意力
        # win_index = 1
        # patch_size = 3 ^ 2 * win_index ^ 2
        self.conv1 = AttentionConv(3, 6, 1, 1)
        # 创建指数级增长的卷积核注意力

    def forward(self, x):
        print(x.shape)
        x1 = self.conv1(x)
        print(x1.shape)
        return x + x1


class AttentionConv(nn.Module):
    def __init__(self, in_channel, window_size, qk_scale=1, num_heads=1):
        super().__init__()
        assert window_size % 3 == 0  # win_size需为三的倍数
        self.in_channel = in_channel
        self.window_size = window_size
        self.qk_scale = qk_scale
        self.patch_size = patch_size = window_size // 3
        self.num_heads = num_heads
        head_channel = in_channel // num_heads

        # 相对位置表，非学习参数
        # 典型的两类
        # 上下左右中：下patch
        down_patch_position = torch.zeros(patch_size, patch_size).cuda(cuda_index)
        for i in range(patch_size):
            down_patch_position[i, :] = np.exp(- i / (patch_size / 2))  # 此处2为超参数
        # print(down_patch_position)
        # up_patch_position = torch.flip(down_patch_position, dims=[0])
        right_patch_position = down_patch_position.transpose(1, 0)
        # left_patch_position = torch.flip(right_patch_position, dims=[1])
        # 四角：右下角patch
        bottom_right_patch_position = torch.zeros(patch_size, patch_size).cuda(cuda_index)
        for i in range(patch_size):
            for j in range(i + 1):
                bottom_right_patch_position[i][j] = np.exp(- (i + j) / (patch_size / 2))
        for i in range(patch_size):
            for j in range(i + 1, patch_size):
                bottom_right_patch_position[i][j] = bottom_right_patch_position[j][i]
        # print(bottom_right_patch_position)
        # merge_position = torch.ones(window_size, window_size)
        # # print(merge_position[:patch_size][patch_size:2*patch_size])
        # merge_position[:patch_size, patch_size:2 * patch_size] = torch.flip(down_patch_position, dims=[0])
        # merge_position[2 * patch_size:, patch_size:2 * patch_size] = down_patch_position
        # merge_position[patch_size:2 * patch_size, :patch_size] = torch.flip(right_patch_position, dims=[1])
        # merge_position[patch_size:2 * patch_size, 2 * patch_size:] = right_patch_position
        #
        # merge_position[2 * patch_size:, 2 * patch_size:] = bottom_right_patch_position
        # merge_position[2 * patch_size:, :patch_size] = torch.rot90(bottom_right_patch_position, k=3, dims=(0, 1))
        # merge_position[:patch_size, :patch_size] = torch.rot90(bottom_right_patch_position, k=2, dims=(0, 1))
        # merge_position[:patch_size, 2 * patch_size:] = torch.rot90(bottom_right_patch_position, k=1, dims=(0, 1))
        #
        # print(f'merge:{merge_position}')

        merge_position = torch.cat(
            [torch.rot90(bottom_right_patch_position, k=2, dims=(0, 1)).unsqueeze(0),
             torch.flip(down_patch_position, dims=[0]).unsqueeze(0),
             torch.rot90(bottom_right_patch_position, k=1, dims=(0, 1)).unsqueeze(0),
             torch.flip(right_patch_position, dims=[1]).unsqueeze(0),
             torch.ones(patch_size, patch_size).unsqueeze(0).cuda(cuda_index),
             right_patch_position.unsqueeze(0),
             torch.rot90(bottom_right_patch_position, k=3, dims=(0, 1)).unsqueeze(0),
             down_patch_position.unsqueeze(0),
             bottom_right_patch_position.unsqueeze(0)
             ], dim=0
        ).cuda(cuda_index)
        # print(merge_position.shape)
        # print(merge_position.view(9,-1))
        self.relative_position_bias = merge_position
        self.qkv = nn.Linear(in_channel * patch_size * patch_size, 3 * in_channel * patch_size * patch_size, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x:input shape:(batch, channel, H, W) 典型：4*3*400*400
        # 图像分割, window_size为3的倍数, kqv对应的是一个window
        b, c, h, w = x.shape
        # print(x)
        padding_h = h - self.patch_size * h // self.patch_size
        padding_w = w - self.patch_size * w // self.patch_size
        x = F.pad(x, (padding_h, padding_h, padding_w, padding_w), mode='constant')
        # print(x)
        # 类似卷积核
        x1 = torch.unfold_copy(x, 2, self.window_size, self.patch_size)  # 沿h维度展开
        x1 = torch.unfold_copy(x1, 3, self.window_size, self.patch_size)  # 沿w维度展开
        # x1 = torch.unfold_copy(x, 2, self.window_size, self.patch_size)  # 沿h维度展开
        # x1 = torch.unfold_copy(x1, 3, self.window_size, self.patch_size)  # 沿w维度展开
        # print(x)
        num_h = x1.shape[2]
        num_w = x1.shape[3]
        # print(x1.shape)  # b*c*num_h*num_w*window_size*window_size
        x1 = x1.permute(0, 2, 3, 1, 4, 5).contiguous()
        x1 = x1.view(-1, self.in_channel, self.window_size, self.window_size)
        # print(x1)
        # print(x1.shape)  # (b*num_h*num_w)*c*window_size*window_size

        # 将窗口分块3*3,每块patch*patch
        x1 = x1.view(-1, self.in_channel, 3, self.patch_size, 3, self.patch_size).permute(0, 2, 4, 1, 3, 5).contiguous()
        # print(f'w {x1.shape}')
        x1 = x1.view(-1, 9, self.in_channel, self.patch_size, self.patch_size).view(-1, 9, self.in_channel * self.patch_size * self.patch_size)
        # print(x1.shape)
        qkv = self.qkv(x1)
        # print(qkv.shape)
        if self.num_heads > 1:
            qkv = qkv.reshape(-1, self.num_heads, self.in_channel // self.num_heads, self.window_size * self.window_size).permute(2, 0, 3, 1, 4)
        qkv = qkv.view(-1, 9, 3, self.in_channel, self.patch_size, self.patch_size).permute(0, 3, 2, 1, 4,
                                                                                            5)  # 将channel放置到前面  # .contiguous().view(-1, )
        # print(qkv.shape)
        # q, k, v = qkv[:,:,0,:,:,:], qkv[:,:,1,:,:,:], qkv[:,:,2,:,:,:]
        q, k, v = torch.unbind(qkv, dim=2)
        # print(q.shape)

        # 对q做处理
        # 选择第三维的第一个矩阵
        # 0 1 2
        # 3 4 5
        # 6 7 8
        core_q = q[:, :, 4, :, :]
        # print(core_q)
        # 使用 repeat 方法复制第一个矩阵，并覆盖整个第三维的数据
        q = core_q.unsqueeze(2).repeat(1, 1, 9, 1, 1)
        # print(q)
        # print(q.shape)

        q = q * self.qk_scale

        # attention_score = torch.sum(torch.mul(torch.mul(q, k), self.relative_position_bias), dim=[-1, -2])
        attention_score = torch.mul(torch.mul(q, k), self.relative_position_bias)

        # print(attention_score.shape)
        # 是否考虑softmax， 后续看看二维softmax该如何设计
        attention_weight = attention_score
        # attention_weight = self.softmax(attention_score)
        # print(v.shape)
        attention_output = torch.sum(torch.mul(attention_weight, v), dim=2)
        # print(attention_output.shape)
        # attention_output = torch.randn(4, 3, 2, 2)
        # print(attention_output)
        attention_output = attention_output.view(b, -1, 3, self.patch_size, self.patch_size).permute(0, 2, 1, 3, 4).contiguous().view(b, 3, num_h,
                                                                                                                                      num_w,
                                                                                                                                      self.patch_size,
                                                                                                                                      self.patch_size).permute(
            0, 1, 2, 4, 3, 5).contiguous().view(b, 3, num_h * self.patch_size, -1)
        # print(attention_output.shape)
        # print(attention_output)
        return attention_output
