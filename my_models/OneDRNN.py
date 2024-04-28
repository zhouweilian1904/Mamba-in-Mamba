# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch
from torch.nn import init
import numpy as np
# from multiscan_v1 import multiscan_v1
# from multiscan_v2 import multiscan_v2

from einops import rearrange


class OneDRNN(nn.Module):
    """
    one direction rnn with spatial consideration which has a patch size
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.LSTM)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(OneDRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.lstm_seq = nn.LSTM(input_channels, patch_size ** 2, 1)
        self.gru_2 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_1 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_2 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_4 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_5 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_6 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_7 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_8 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d((patch_size ** 2) * input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.fc = nn.Linear((patch_size ** 2) * input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size ** 2), n_classes)
        self.rec_2 = nn.Linear((patch_size ** 2) * (patch_size ** 2), input_channels)
        self.pooling = nn.MaxPool2d(3, 3)
        self.dropout = nn.Dropout(0.1)
        self.aux_loss_weight = 1

    def forward(self, x):  # 初始是第三方向
        # print("x",x.shape)
        x = x.squeeze(1)
        # 生成第一方向
        x1 = torch.transpose(x, 2, 3)
        x1r = x.reshape(x1.shape[0], x1.shape[1], -1)
        # print('0', x1r.shape)
        # 生成第二方向
        x2 = x1r.cpu()
        x2rn = np.flip(x2.numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()
        # print('2',x.shape)
        # 生成第三方向
        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)
        # # #生成第四方向 从第三方向来
        x4 = x3r.cpu()
        x4rn = np.flip(x4.numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()
        # 生成第五方向
        x5 = torch.rot90(x1, 1, (2, 3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)
        # #生成第六方向
        x6 = x5r.cpu()
        x6rn = np.flip(x6.numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()
        # #生成第七方向
        x7 = torch.transpose(x5, 2, 3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)
        # #生成第八方向
        x8 = x7r.cpu()
        x8rn = np.flip(x8.numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()
        # 下面改变输入值，确定使用哪个方向
        x1r = x1r.permute(2, 0, 1)
        x2r = x2r.permute(2, 0, 1)
        x3r = x3r.permute(2, 0, 1)
        x4r = x4r.permute(2, 0, 1)
        x5r = x5r.permute(2, 0, 1)
        x6r = x6r.permute(2, 0, 1)
        x7r = x7r.permute(2, 0, 1)
        x8r = x8r.permute(2, 0, 1)  # s  b  c

        # x_1 = multiscan(rearrange(x, 'b c h w -> b h w c'), 1)
        # multi_2 = multiscan(rearrange(x, 'b c w h -> b h w c'), 2)
        # multi_3 = multiscan(rearrange(x, 'b c w h -> b h w c'), 3)
        # multi_4 = multiscan(rearrange(x, 'b c w h -> b h w c'), 4)
        # print('multi', multi_1.shape) #multi torch.Size([100, 103, 25])

        # 导入GRU
        # x = self.gru(x1r+x2r)[0]
        x1 = self.gru_2_1(x1r)
        x2 = self.gru_2_2(x2r)
        x3 = self.gru_2_3(x3r)
        x4 = self.gru_2_4(x4r)
        x5 = self.gru_2_5(x5r)
        x6 = self.gru_2_6(x6r)
        x7 = self.gru_2_7(x7r)
        x8 = self.gru_2_8(x8r)

        x_1 = x1
        x_2 = x2
        x_3 = x3
        x_4 = x4
        x_5 = x5
        x_6 = x6
        x_7 = x7
        x_8 = x8

        x_1 = x_1.permute(1, 2, 0).contiguous()
        x_2 = x_2.permute(1, 2, 0).contiguous()
        x_3 = x_3.permute(1, 2, 0).contiguous()
        x_4 = x_4.permute(1, 2, 0).contiguous()
        x_5 = x_5.permute(1, 2, 0).contiguous()
        x_6 = x_6.permute(1, 2, 0).contiguous()
        x_7 = x_7.permute(1, 2, 0).contiguous()
        x_8 = x_8.permute(1, 2, 0).contiguous()
        # print(x.shape) #(16,64,103)
        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        x_5 = x_5.view(x_5.size(0), -1)
        x_6 = x_6.view(x_6.size(0), -1)
        x_7 = x_7.view(x_7.size(0), -1)
        x_8 = x_8.view(x_8.size(0), -1)
        # print('5',x.shape) #(16,6592)
        # plt.subplot(3,2,2)
        # plt.plot(x[-1,:].cpu().detach().numpy())
        # x = self.gru_bn(x)
        x_1 = self.gru_bn_2(x_1)
        x_2 = self.gru_bn_2(x_2)
        x_3 = self.gru_bn_2(x_3)
        x_4 = self.gru_bn_2(x_4)
        # plt.subplot(3, 2, 3)
        # plt.plot(x[-1, :].cpu().detach().numpy())
        x_1 = self.relu(x_1)
        x_2 = self.relu(x_2)
        x_3 = self.relu(x_3)
        x_4 = self.relu(x_4)
        # plt.subplot(3, 2, 4)
        # plt.plot(x[-1, :].cpu().detach().numpy())
        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)
        x_3 = self.dropout(x_3)
        x_4 = self.dropout(x_4)
        # plt.subplot(3, 2, 5)
        # plt.plot(x[-1, :].cpu().detach().numpy())
        # x = self.fc(x)
        x_class = self.fc_2(x_1)
        x_2class = self.fc_2(x_2)
        x_3class = self.fc_2(x_3)
        x_4class = self.fc_2(x_4)
        x_rec = self.rec_2(x_1)
        return x_class+ x_2class +x_3class +x_4class, x_rec
