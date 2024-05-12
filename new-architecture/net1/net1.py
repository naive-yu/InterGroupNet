import numpy as np
import torch
import torch.nn as nn
# import math
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

torch.set_printoptions(profile="full")
cuda_index = 1


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
        # 创建指数级增长的卷积核注意力 PID 18236

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
        # 典型的两类
        # 上下左右中：下patch
        down_patch_position = torch.zeros(patch_size, patch_size).cuda(cuda_index)
        for i in range(patch_size):
            down_patch_position[i, :] = np.exp(- i / (patch_size / pos_decay))  # 此处pos_decay为参数
        # print(down_patch_position)
        # up_patch_position = torch.flip(down_patch_position, dims=[0])
        right_patch_position = down_patch_position.transpose(1, 0)
        # left_patch_position = torch.flip(right_patch_position, dims=[1])
        # 四角：右下角patch
        bottom_right_patch_position = torch.zeros(patch_size, patch_size).cuda(cuda_index)
        for i in range(patch_size):
            for j in range(i + 1):
                bottom_right_patch_position[i][j] = np.exp(- (i + j) / (patch_size / pos_decay))  # 此处pos_decay为参数
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
        ).view(9, -1).cuda(cuda_index)
        print(merge_position)
        # print(merge_position.view(9,-1))
        linear_dim = in_channel * patch_size * patch_size
        self.sqrt_constant = torch.sqrt(torch.tensor(linear_dim, dtype=torch.float32).cuda(cuda_index))
        self.relative_position_bias = merge_position
        # self.qkv = nn.Linear(in_channel * patch_size * patch_size, 3 * in_channel * patch_size * patch_size, bias=True)
        self.q_s = nn.ModuleList([nn.Linear(linear_dim, linear_dim, bias=True) for _ in range(num_heads)]).cuda(cuda_index)
        self.k_s = nn.ModuleList([nn.Linear(linear_dim, linear_dim, bias=True) for _ in range(num_heads)]).cuda(cuda_index)
        self.v_s = nn.ModuleList([nn.Linear(linear_dim, linear_dim, bias=True) for _ in range(num_heads)]).cuda(cuda_index)
        self.softmax = nn.Softmax(dim=-1).cuda(cuda_index)
        self.head_weight = nn.Parameter(torch.ones(1, num_heads)/num_heads, requires_grad=True).cuda(cuda_index)

    def forward(self, x1):
        # print("att_forward")
        # x:input shape:(batch, channel, H, W) 典型：4*3*400*400
        # 图像分割, window_size为3的倍数, kqv对应的是一个window
        # x1 = x
        batch, channel, height, width = x1.shape
        # assert channel == self.in_channel  # 通道数一致
        # print(x1)
        # h,w不整除patch_size，默认放在上侧和左侧
        padding_h = (self.patch_size * (1 + height // self.patch_size) - height) % self.patch_size
        padding_w = (self.patch_size * (1 + width // self.patch_size) - width) % self.patch_size
        # print(f'padding_w{padding_w}, padding_h{padding_h}')
        # 边界patch填充
        x1 = F.pad(x1, (self.patch_size + padding_w, self.patch_size, self.patch_size + padding_h, self.patch_size), mode='reflect')
        # print(x1.shape)
        # 类似卷积核
        x1 = x1.unfold(2, self.window_size, self.patch_size).contiguous()  # 沿h维度展开
        x1 = x1.unfold(3, self.window_size, self.patch_size).contiguous()  # 沿w维度展开
        # print(f'unfold{x1.shape}')
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
        # x1 = x1.view(-1, 9, self.in_channel, self.patch_size * self.patch_size).view(-1, 9, self.in_channel * self.patch_size * self.patch_size)
        x1 = x1.view(-1, 9, self.in_channel * self.patch_size * self.patch_size)
        # print(x1.shape)
        q_s = torch.cat([q(x1).unsqueeze(1) for q in self.q_s], dim=1)
        k_s = torch.cat([k(x1).unsqueeze(1) for k in self.k_s], dim=1).transpose(-2, -1).contiguous()
        v_s = torch.cat([v(x1).unsqueeze(1) for v in self.v_s], dim=1)
        # print(q_s.shape)
        
        # qkv = qkv.view(-1, 9, 3, self.in_channel, self.patch_size * self.patch_size).permute(0, 3, 2, 1, 4)
        # (b*windows_num)*9*(qkv)*in_channel*patch_size*patch_size
        # if self.num_heads > 1:
        #     qkv = qkv.view(-1, self.num_heads, self.in_channel // self.num_heads, self.patch_size * self.patch_size).permute(2, 0, 3, 1, 4)
        # 将channel放置到前面  # .contiguous().view(-1, )
        # 32760*9*3*3*6*6 -> 32760*C(3)*3*9*6*6
        # print(qkv.shape)
        # q, k, v = qkv[:,:,0,:,:,:], qkv[:,:,1,:,:,:], qkv[:,:,2,:,:,:]
        # q, k, v = torch.unbind(qkv, dim=2)
        # print(q.shape)

        # 对q做处理
        # 选择第三维的第四个矩阵
        # 0 1 2
        # 3 4 5
        # 6 7 8
        core_q = q_s[:, :, 4, :].unsqueeze(2).contiguous()
        # print(core_q)
        # 使用 repeat 方法复制第一个矩阵，并覆盖整个第三维的数据
        # q_s = core_q.unsqueeze(2).repeat(1, 1, 9, 1)
        # print(q)
        # print(q.shape)

        # core_q = core_q * self.qk_scale

        # attention_score = torch.sum(torch.mul(torch.mul(q, k), self.relative_position_bias), dim=[-1, -2])
        # attention_score = torch.mul(torch.mul(q, k), self.relative_position_bias)

        # attention_score = torch.mul(q_s, k_s)
        # print(f'cq{core_q.shape}')
        # print(f'ks{k_s.shape}')
        
        attention_score = torch.matmul(core_q, k_s)
        attention_score = attention_score.div_(self.sqrt_constant)
        # print(f'as{attention_score.shape}') # 76800*3*9*2*2
        # 是否考虑softmax， 后续看看softmax该如何设计
        attention_weight = self.softmax(attention_score)
        # print(f'aw{attention_weight.shape}')
        # if self.num_heads != 1:
        #     attention_weight = attention_score
        # print(f'vs{v_s.shape}')
        attention_output = torch.matmul(attention_weight, v_s).squeeze(2)
        # 多头注意力加权
        # print(f'ao{attention_output.shape}')
        attention_output = torch.matmul(self.head_weight, attention_output)
        # print(f'ao2{attention_output.shape}')
        # attention_output = torch.randn(4, 3, 2, 2)
        # print(attention_output)
        attention_output = attention_output.view(batch, -1, self.in_channel, self.patch_size, self.patch_size).permute(0, 2, 1, 3, 4).contiguous()
        attention_output = attention_output.view(batch, self.in_channel, num_h, num_w, self.patch_size, self.patch_size).permute(0, 1, 2, 4, 3, 5).contiguous()
        attention_output = attention_output.view(batch, self.in_channel, num_h * self.patch_size, -1)
        # print(f'ao3{attention_output.shape}')
        # 裁剪恢复图像 padding值可能为0
        # end_h = attention_output.shape[-2] - padding_h
        # end_w = attention_output.shape[-1] - padding_w
        attention_output = attention_output[:, :, padding_h:, padding_w:]

        # print(attention_output)
        return attention_output


# my_conv1 = AttentionConv(3, 9, num_heads=2).cuda(cuda_index)
# my_input = torch.randn(1, 3, 40, 40, requires_grad=True).cuda(cuda_index)
# # # my_conv1 = AttentionConv(3, 3, 1)
# # # my_input = torch.randn(1, 3, 4, 4, requires_grad=True)
# # print(my_input.shape)
# output = my_conv1(my_input)
