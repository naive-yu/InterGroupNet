import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

torch.set_printoptions(profile="full")
cuda_index = 2


class DehazeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        # 创建指数级增长的卷积核注意力
        # win_index = 1
        # patch_size = 3 ^ 2 * win_index ^ 2
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        # self.conv18 = AttentionConv(3, 18)
        # self.conv15 = AttentionConv(3, 15)
        # self.conv12 = AttentionConv(3, 12)
        self.conv9 = AttentionConv(3, 9, num_heads=2)
        self.conv6 = AttentionConv(3, 6, num_heads=4)
        self.conv3 = AttentionConv(3, 3, num_heads=6)
        self.conv0 = nn.Conv2d(9, 3, 5, 1, 2)
        # 创建指数级增长的卷积核注意力

    def forward(self, x):
        b, c, h, w = x.shape
        # print(x.shape)
        # 2 * 3 * 480 * 640
        x_g = self.softmax(self.global_avg_pool((1 - x).view(b, c, -1))).unsqueeze(-1)
        # print(x_g.shape)  # 2 * 3 * 1 * 1
        x_g = self.global_max_pool((x * x_g).reshape(b, 1, -1)).unsqueeze(-1)
        # print(x_g.shape)

        # x18 = self.relu(self.conv18(x))
        # # print(x18.shape)
        # x12 = self.relu(self.conv12(x18))
        # # print(x12.shape)
        # x9 = self.relu(self.conv9(x12))
        # # print(x9.shape)
        # x6 = self.relu(self.conv6(x9))
        # # print(x6.shape)
        # x3 = self.relu(self.conv3(x6))
        # print(x3.shape)  # 训练后期出现问题

        # x3 = self.relu(self.conv3(x))
        # # print(x3.shape)
        # x6 = self.relu(self.conv6(x3))
        # # print(x6.shape)
        # x9 = self.relu(self.conv9(x6))
        # # print(x9.shape)
        # x12 = self.relu(self.conv12(x9))
        # # print(x12.shape)
        # x15 = self.relu(self.conv15(x12))
        # # print(x15.shape)
        # # x18 = self.relu(self.conv18(x15))
        # # print(x18.shape)
        # x0 = self.relu(self.conv0(torch.cat([x15, x12, x9, x6, x3], dim=1)))

        x3 = self.conv3(x)
        # print(x3.shape)
        x6 = self.conv6(x3)
        # print(x6.shape)
        x9 = self.conv9(x6)
        # print(x9.shape)
        # x12 = self.conv12(x9)
        # print(x12.shape)
        # x15 = self.conv15(x9)
        # print(x15.shape)
        # x18 = self.relu(self.conv18(x15))
        # print(x18.shape)
        x0 = self.relu(self.conv0(torch.cat([x9, x6, x3], dim=1)))

        # print(x0.shape)
        # return self.relu(x * x0 + x0*(1 - x0))
        return self.relu((x * x0) + (x_g - x0))
        # return self.relu(x * x0)  # 很烂


class AttentionConv(nn.Module):
    def generate_position(self, patch_size, pos_decay):
        # 上下左右：右patch
        right_patch_position = torch.exp(-torch.arange(patch_size, dtype=torch.float32) / (patch_size / pos_decay)).expand(patch_size, -1).cuda(cuda_index)
        # print(right_patch_position)
        down_patch_position = right_patch_position.transpose(-2, -1)
        # 四角：右下角patch
        i, j = torch.meshgrid(torch.arange(patch_size, dtype=torch.float32), torch.arange(patch_size, dtype=torch.float32))
        bottom_right_patch_position = torch.exp(-(i + j) / (patch_size / pos_decay)).cuda(cuda_index)

        # 将四个位置张量按照顺序拼接
        merge_position = torch.stack([
            torch.rot90(bottom_right_patch_position, k=2, dims=(0, 1)),
            down_patch_position.flip(0),
            torch.rot90(bottom_right_patch_position, k=1, dims=(0, 1)),
            right_patch_position.flip(1),
            torch.ones(patch_size, patch_size).cuda(cuda_index),
            right_patch_position,
            torch.rot90(bottom_right_patch_position, k=3, dims=(0, 1)),
            down_patch_position,
            bottom_right_patch_position
        ], dim=0).view(9, -1)
        # print(merge_position)
        return merge_position
    
    
    def __init__(self, in_channel, window_size, num_heads=1, qk_scale=1, pos_decay=1):
        super().__init__()
        assert window_size % 3 == 0  # win_size需为三的倍数
        self.in_channel = in_channel
        self.window_size = window_size
        self.qk_scale = qk_scale
        self.patch_size = patch_size = window_size // 3
        self.num_heads = num_heads
        # self.dropout = torch.dropout()

        # 相对位置表，非学习参数
        self.relative_position_bias = self.generate_position(patch_size, pos_decay)
        # result_shape: [num_patch*num_patch, patch_size*patch_size]
        linear_dim = in_channel * patch_size * patch_size
        self.sqrt_constant = torch.sqrt(torch.tensor(linear_dim, dtype=torch.float32))
        self.qkv_s = nn.Linear(linear_dim, num_heads * 3 * linear_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.head_weight = nn.Parameter(torch.ones(1, num_heads)/num_heads, requires_grad=True)

    def forward(self, x):
        # x:input shape:[batch, channel, h, w] 
        # 训练图像：4*3*400*400/4*3*480*640
        # 图像分割, window_size为3的倍数, 多个kqv对应的是一个window
        x1 = x
        batch, channel, height, width = x1.shape
        # h,w不整除patch_size，默认放在上侧和左侧
        padding_h = (self.patch_size * (1 + height // self.patch_size) - height) % self.patch_size
        padding_w = (self.patch_size * (1 + width // self.patch_size) - width) % self.patch_size
        # 边界patch填充
        x1 = F.pad(x1, (self.patch_size + padding_w, self.patch_size, self.patch_size + padding_h, self.patch_size), mode='reflect')
        # 类似卷积核
        x1 = x1.unfold(2, self.window_size, self.patch_size)  # 沿h维度展开
        x1 = x1.unfold(3, self.window_size, self.patch_size)  # 沿w维度展开
        # result_shape: [batch, channel, num_h, num_w, window_size, window_size]
        num_h = x1.shape[2]
        num_w = x1.shape[3]
        x1 = x1.permute(0, 2, 3, 1, 4, 5).contiguous()
        # result_shape: [batch, num_h, num_w, channel, window_size, window_size]
        # x1 = x1.view(-1, self.in_channel, self.window_size, self.window_size)
        # result_shape: [(batch*num_h*num_w), channel, window_size, window_size]

        # 将窗口分块3*3,每块patch*patch
        x1 = x1.view(-1, self.in_channel, 3, self.patch_size, 3, self.patch_size).permute(0, 2, 4, 1, 3, 5).contiguous()
        # result_shape: [(batch*num_h*num_w), num_patch, num_patch, channel, patch_size, patch_size]
        x1 = x1.view(-1, 9, self.in_channel * self.patch_size * self.patch_size)
        # result_shape: [(batch*num_h*num_w), (num_patch*num_patch), (channel*patch_size*patch_size)]
        qkv = self.qkv_s(x1)
        # result_shape: [(batch*num_h*num_w), (num_patch*num_patch), (num_heads*3*channel*patch_size*patch_size)]
        if self.num_heads > 1:  # 多头注意力
            qkv = qkv.view(-1, 9, self.num_heads, 3, self.in_channel, self.patch_size * self.patch_size).permute(0, 2, 4, 3, 1, 5).contiguous()
            # result_shape_first: [(batch*num_h*num_w), (num_patch*num_patch), num_heads, 3, channel, (patch_size*patch_size)]
            # result_shape_second: [(batch*num_h*num_w), num_heads, channel, 3, (num_patch*num_patch), (patch_size*patch_size)]
            qkv = qkv.view(-1, self.in_channel, 3, 9, self.patch_size * self.patch_size)
            # result_shape: [(batch*num_h*num_w*num_heads), channel, 3, (num_patch*num_patch), (patch_size*patch_size)]
        else:
            qkv = qkv.view(-1, 9, 3, self.in_channel, self.patch_size * self.patch_size).permute(0, 3, 2, 1, 4)
            # result_shape_first: [(batch*num_h*num_w), (num_patch*num_patch), 3, channel, (patch_size*patch_size)]
            # result_shape_second: [(batch*num_h*num_w), channel, 3, (num_patch*num_patch), (patch_size*patch_size)]
        # q, k, v = qkv[:,:,0,:,:], qkv[:,:,1,:,:], qkv[:,:,2,:,:]
        q, k, v = torch.unbind(qkv, dim=2)
        # result_shape: [(batch*num_h*num_w*num_heads), channel, (num_patch*num_patch), (patch_size*patch_size)]
        
        # 对q做处理，选择第三维的第五个矩阵
        # 0 1 2
        # 3 4 5
        # 6 7 8
        core_q = q[:, :, 4, :]
        # 添加第三个维度
        q = core_q.unsqueeze(2)
        # result_shape: [(batch*num_h*num_w*num_heads), channel, 1, (patch_size*patch_size)]
        # q = q * self.qk_scale
        k = torch.mul(k, self.relative_position_bias).transpose(-2,-1)
        # [(batch*num_h*num_w*num_heads), channel, (num_patch*num_patch), (patch_size*patch_size)]*[num_patch*num_patch, patch_size*patch_size]
        # result_shape: [(batch*num_h*num_w*num_heads), channel, (patch_size*patch_size), (num_patch*num_patch)]
        attention_score = torch.matmul(q, k)
        # [(batch*num_h*num_w*num_heads), channel, 1, (patch_size*patch_size)]*[(batch*num_h*num_w*num_heads), channel, (patch_size*patch_size), (num_patch*num_patch)]
        # result_shape: [(batch*num_h*num_w*num_heads), channel, 1, (num_patch*num_patch)]
        attention_score = attention_score.div_(self.sqrt_constant)
        # result_shape: [(batch*num_h*num_w*num_heads), channel, 1, (num_patch*num_patch)]
        attention_weight = self.softmax(attention_score)
        # result_shape: [(batch*num_h*num_w*num_heads), channel, 1, (num_patch*num_patch)]
        attention_output = torch.matmul(attention_weight, v).view(-1, self.num_heads, self.in_channel * self.patch_size * self.patch_size)
        # [(batch*num_h*num_w*num_heads), channel, 1, (num_patch*num_patch)]*[(batch*num_h*num_w*num_heads), channel, (num_patch*num_patch), (patch_size*patch_size)]
        # result_shape_first: [(batch*num_h*num_w*num_heads), channel, 1, (patch_size*patch_size)]
        # result_shape_second: [(batch*num_h*num_w), num_heads, (channel*patch_size*patch_size)]
        attention_output = torch.matmul(self.head_weight, attention_output)
        # [1, num_heads]*[(batch*num_h*num_w), num_heads, (channel*patch_size*patch_size)]
        # result_shape: [(batch*num_h*num_w), 1, (channel*patch_size*patch_size)]
        attention_output = attention_output.view(batch, num_h, num_w, self.in_channel, self.patch_size, self.patch_size).permute(0, 3, 1, 2, 4, 5).contiguous()
        # result_shape_first: [batch, num_h, num_w, channel, patch_size, patch_size]
        # result_shape_second: [batch, channel, num_h, num_w, patch_size, patch_size]
        attention_output = attention_output.permute(0, 1, 2, 4, 3, 5).contiguous()
        # result_shape: [batch, channel, num_h, patch_size, num_w, patch_size]
        attention_output = attention_output.view(batch, self.in_channel, num_h * self.patch_size, -1)
        # result_shape: [batch, channel, (num_h*patch_size), (num_w*patch_size)]
        # 裁剪恢复图像 padding值可能为0
        attention_output = attention_output[:, :, padding_h:, padding_w:]
        # result_shape: [batch, channel, h, w]
        return attention_output


# my_conv1 = AttentionConv(3, 9, num_heads=2).cuda(cuda_index)
# my_input = torch.randn(1, 3, 40, 40, requires_grad=True).cuda(cuda_index)
# # # my_conv1 = AttentionConv(3, 3, 1)
# # # my_input = torch.randn(1, 3, 4, 4, requires_grad=True)
# # print(my_input.shape)
# output = my_conv1(my_input)
