# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch
from torch.nn import init
import numpy as np

class FourDRNN(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(FourDRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_2_1 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_2_2 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_2_3 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_2_4 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_2_5 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_2_6 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_2_7 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_2_8 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.gru_3 = nn.LSTM(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * 64)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_3 = nn.BatchNorm1d(64 * (patch_size**2))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size**2 * 64 , n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.fc_3 = nn.Linear(64 * (patch_size**2), n_classes)
        self.rec_3 = nn.Linear(64 * (patch_size ** 2), input_channels)
        self.softmax = nn.Softmax()
        self.aux_loss_weight = 1

    def forward(self, x): #初始是第1方向
        x = x.squeeze(1)
        # print('0', x.shape)
        x1 = x
        x1r = x1.reshape(x1.shape[0], x1.shape[1], -1)
        # plt.plot(x1r[0,:,:].cpu().detach().numpy())
        # plt.plot(x1r[0,:,12].cpu().detach().numpy(), linewidth=5)
        # plt.show()

        # x2 = Variable(x1r.cpu())
        # x2 = Variable(x1r).cpu()
        x2 = x1r.cpu()
        x2rn = np.flip(x2.numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()

        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)

        # x4 = Variable(x3r.cpu())
        # x4 = Variable(x3r).cpu()
        x4 = x3r.cpu()
        x4rn = np.flip(x4.numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()

        x5 = torch.rot90(x1, 1, (2, 3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)

        # x6 = Variable(x5r.cpu())
        # x6 = Variable(x5r).cpu()
        x6 = x5r.cpu()
        x6rn = np.flip(x6.numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()

        x7 = torch.transpose(x5, 2, 3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)

        # x8 = Variable(x7r.cpu())
        # x8 = Variable(x7r).cpu()
        x8 = x7r.cpu()
        x8rn = np.flip(x8.numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()
        # print('x8r',x8r.shape) #(16,103,25)
        # plt.plot(x1r[0,:,:].cpu().detach().numpy())

        x8r = x8r.permute(2, 0, 1) #(25,16,103)
        x7r = x7r.permute(2, 0, 1)
        x6r = x6r.permute(2, 0, 1)
        x5r = x5r.permute(2, 0, 1)
        x4r = x4r.permute(2, 0, 1)
        x3r = x3r.permute(2, 0, 1)
        x2r = x2r.permute(2, 0, 1)
        x1r = x1r.permute(2, 0, 1)

        # plt.figure(figsize=(7, 6))
        # plt.plot(x1r[1, 0, :].cpu().numpy(), lw=3.5, color='Red')
        # plt.title('Spectral signal of one pixel', fontsize=20)
        # plt.xticks(np.arange(1, 201, 19).astype(int), fontsize=20)  # Set x-axis ticks from 1 to 200
        # plt.ylabel('Spectral Value', fontsize=20)
        # plt.yticks(fontsize=20)  # Increase y-axis tick label font size to 12
        # plt.xlabel('Band Number', fontsize=20)
        # # Adding the bounding box
        # ax = plt.gca()
        # box = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=None, edgecolor='black', linewidth=5)
        # ax.add_patch(box)
        # plt.show()

        # print('x8r', x8r.shape) #(25,16,103)
        # plt.subplot(1,5,1)
        # plt.subplot(1, 5, 2)
        # plt.plot(x2r[:, 0, :].cpu().detach().numpy())
        # x1r_r = self.gru_2_1(x1r)
        # x1r_r = self.relu(x1r_r)
        # print('x1r',x1r.shape) #(25,16,103)
        # plt.subplot(1,5,1)
        # plt.plot(x1r[:, 0, :].cpu().detach().numpy())
        # plt.subplot(1,5,4)
        # plt.plot((x1r+x2r)[:, 0, :].cpu().detach().numpy())
        # x2r_r = self.gru_2_2(x2r + x1r_r) #把x1r经过RNN的值，作为x2r的输入
        # x2r_r = self.relu(x2r_r)
        # plt.subplot(1,5,2)
        # plt.plot(x2r[:, 0, :].cpu().detach().numpy())
        x3r_r = self.gru_2_3(x3r)
        x3r_r = self.relu(x3r_r)
        # plt.subplot(1, 5, 3)
        # plt.plot(x3r[:, 0, :].cpu().detach().numpy())
        # x4r_r = self.gru_2_4(x4r + x3r_r)
        # x4r_r = self.relu(x4r_r)
        # plt.subplot(1, 5, 4)
        # plt.plot(x4r[:, 0, :].cpu().detach().numpy())
        # x5r_r = self.gru_2_5(x5r + x4r_r)
        # x5r_r = self.relu(x5r_r)
        #
        # x6r_r = self.gru_2_6(x6r + x5r_r)
        # x6r_r = self.relu(x6r_r)
        #
        # x7r_r = self.gru_2_7(x7r + x6r_r)
        # x7r_r = self.relu(x7r_r)
        #
        # x8r_r = self.gru_2_8(x8r)
        # x8r_r = self.relu(x8r_r)
        # x8r = self.gru(x8r+x7r)[0]
        x = x3r_r
        # print(x.shape)
        # x = self.gru_3(x)
        # x = self.gru_2(x)[0]
        # x = torch.cat([x1r,x2r,x3r,x4r,x5r,x6r,x7r,x8r],dim=2)
        # x = self.gru_bn(x)
        # x = x1r + x2r + x3r + x4r + x5r + x6r + x7r + x8r
        # print('x',x.shape)
        # print('into GRU',x3.shape)
        # x4 = self.gru(x4)[0]
        # x3 = self.gru(x3)[0]
        # x2 = self.gru(x2)[0]

        # x = self.gru(x)[0]
        # x = self.gru2(x)[0]

        # print('out GRU',x3.shape)
        # x4 = x4.permute(1, 2, 0).contiguous()
        # x3 = x3.permute(1, 2, 0).contiguous()
        # x2 = x2.permute(1, 2, 0).contiguous()
        # x1 = x1.permute(1, 2, 0).contiguous()
        x = x.permute(1,2,0).contiguous()
        # print('5-1',x1.shape)

        # x4 = x4.view(x4.size(0), -1)
        # x3 = x3.view(x3.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        # x1 = x1.view(x1.size(0), -1)
        x = x.view(x.size(0),-1)

        # x = x4 + x3 + x2 + x1
        # # w1 = x1 / x
        # # w2 = x2 / x
        # # w3 = x3 / x
        # # w4 = x4 / x
        # x = 0.35*x1 + 0.35*x2 + 0.15*x3 +0.15*x4
        # # x = w1*x1 + w2*x2 + w3*x3 + w4*x4
        # print('into gru_bn', x.shape)
        # x = self.gru_bn_2(x)
        x = self.gru_bn_3(x)
        # x = self.gru_bn2(x)
        x = self.relu(x)
        # x = self.tanh(x)
        # x = self.elu(x)
        # x =self.prelu(x)
        # print('into fc',x.shape)
        x = self.dropout(x)
        # x = self.fc_2(x)
        x_class = self.fc_3(x)
        # x_rec = self.rec_3(x)
        # x = self.softmax(x)
        # print(x[0,:].cpu().detach().numpy())
        # plt.plot(x[0,:].cpu().detach().numpy())
        # plt.show()
        # plt.grid(linewidth=0.5, color='black')
        # plt.title('Real situation in one patch', fontdict={'size': 40})
        # plt.xlabel('Band Numbers', fontdict={'size': 40}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # plt.show()
        # x = self.fc2(x)
        return x_class