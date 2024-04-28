import torch.nn as nn
import torch
from torch.nn import init
import numpy as np


class EightDLSTM(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values
        # of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.LSTM)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with
        # sigmoid gate activation and PRetanh activation functions for hidden representations
        super(EightDLSTM, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, input_channels, 1, bidirectional=False)
        self.gru_2_1 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_2 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_4 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_5 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_6 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_7 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_8 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.lstm_2_1 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_2 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_3 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_4 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_5 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_6 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_7 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_8 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_1 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_2 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_3 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_4 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_5 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_6 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_7 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_8 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        self.lstm_stra_1 = nn.LSTM(64, 64, 1, bidirectional=False, batch_first=True)
        # self.gru_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_1 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_2 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_4 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_5 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_6 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_7 = nn.GRU(input_channels, patch_size ** 2 , 1, bidirectional=True)
        # self.gru_3_8 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        self.scan_order = Order_Attention(64, 64)

        self.gru_4 = nn.GRU(64, 64, 1)
        self.lstm_4 = nn.LSTM(patch_size ** 2, 64, 1)
        self.conv = nn.Conv2d(input_channels, out_channels=input_channels, kernel_size=(3, 3), stride=(3, 3))
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64) * 1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size) ** 2)
        self.lstm_bn_2 = nn.BatchNorm1d((64) * 8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size ** 2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(8 * 64)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size ** 2), n_classes)
        self.lstm_fc_2 = nn.Linear(64 * 8, n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size ** 2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size ** 2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.fc_laststep = nn.Linear(8 * 64, n_classes)
        self.reg = nn.Linear(64, input_channels)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(64, 64)
        self.aux_loss_weight = 1

    def forward(self, x):  # 初始是第1方向
        # print('x.shape1',x.shape)
        x = x.squeeze(1)
        # print('x.shape2', x.shape)
        # x = self.conv(x)
        # print('x.shape3', x.shape)
        # x_matrix = x[0,:,:,:]
        # x_matrix = x_matrix.cpu()
        # # plt.subplot(331)
        # plt.imshow(x_matrix[0,:,:], interpolation='nearest', origin='upper')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.xlabel('X-Position', fontdict={'size': 25}, fontweight='bold')
        # plt.ylabel('Y-Position', fontdict={'size': 25}, fontweight='bold')
        # plt.title('Values of last dimension in the patch',fontdict={'size': 20}, fontweight='bold')
        # plt.show()

        # 生成第1和7
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        # plt.subplot(3, 4, 9).set_title('Spectral signatures in a patch')
        # direction_1_showpicture = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)
        # plt.xlabel('Band Numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 15},fontweight='bold')
        # plt.plot(direction_1_showpicture[0, :, :].cpu().detach().numpy())
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        direction_1 = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)

        # print('d1',direction_1.shape)
        # print('d1',direction_1.shape)
        direction_7 = torch.flip(direction_1, [2])

        # 生成第2和8
        x2_0 = x[:, :, :, 0]
        x2_1 = x[:, :, :, 1]
        x2_2 = x[:, :, :, 2]
        x2_3 = x[:, :, :, 3]
        x2_4 = x[:, :, :, 4]
        x2_1f = torch.flip(x2_1, [2])
        x2_3f = torch.flip(x2_3, [2])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        direction_2 = torch.cat([x2_0, x2_1f, x2_2, x2_3f, x2_4], dim=2)
        direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        x3_0 = x[:, :, 0, :]
        x3_1 = x[:, :, 1, :]
        x3_2 = x[:, :, 2, :]
        x3_3 = x[:, :, 3, :]
        x3_4 = x[:, :, 4, :]
        x3_0f = torch.flip(x3_0, [2])
        x3_2f = torch.flip(x3_2, [2])
        x3_4f = torch.flip(x3_4, [2])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        direction_3 = torch.cat([x3_0f, x3_1, x3_2f, x3_3, x3_4f], dim=2)
        direction_5 = torch.flip(direction_3, [2])

        # 生成4和6
        x4_0 = x[:, :, :, 0]
        x4_1 = x[:, :, :, 1]
        x4_2 = x[:, :, :, 2]
        x4_3 = x[:, :, :, 3]
        x4_4 = x[:, :, :, 4]
        x4_1f = torch.flip(x4_1, [2])
        x4_3f = torch.flip(x4_3, [2])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # print('d4', direction_4.shape)
        direction_6 = torch.flip(direction_4, [2])

        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_1[0, :, 0].cpu().detach().numpy(), label='index 0')
        # plt.plot(direction_1[0, :, 1].cpu().detach().numpy(), label='index 1')
        # plt.plot(direction_1[0, :, 2].cpu().detach().numpy(), label='index 2')
        # plt.plot(direction_1[0, :, 3].cpu().detach().numpy(), label='index 3')
        # plt.plot(direction_1[0, :, 4].cpu().detach().numpy(), label='index 4')
        # plt.plot(direction_1[0, :, 5].cpu().detach().numpy(), label='index 9')
        # plt.plot(direction_1[0, :, 6].cpu().detach().numpy(), label='index 8')
        # plt.plot(direction_1[0, :, 7].cpu().detach().numpy(), label='index 7')
        # plt.plot(direction_1[0, :, 8].cpu().detach().numpy(), label='index 6')
        # plt.plot(direction_1[0, :, 9].cpu().detach().numpy(), label='index 5')
        # plt.plot(direction_1[0, :, 10].cpu().detach().numpy(), label='index 10')
        # plt.plot(direction_1[0, :, 11].cpu().detach().numpy(), label='index 11')
        # plt.plot(direction_1[0, :, 12].cpu().detach().numpy(), label='index 12', linewidth=5, linestyle='-.', color = 'red' )
        # plt.plot(direction_1[0, :, 13].cpu().detach().numpy(), label='index 13')
        # plt.plot(direction_1[0, :, 14].cpu().detach().numpy(), label='index 14')
        # plt.plot(direction_1[0, :, 15].cpu().detach().numpy(), label='index 19')
        # plt.plot(direction_1[0, :, 16].cpu().detach().numpy(), label='index 18')
        # plt.plot(direction_1[0, :, 17].cpu().detach().numpy(), label='index 17')
        # plt.plot(direction_1[0, :, 18].cpu().detach().numpy(), label='index 16')
        # plt.plot(direction_1[0, :, 19].cpu().detach().numpy(), label='index 15')
        # plt.plot(direction_1[0, :, 20].cpu().detach().numpy(), label='index 20')
        # plt.plot(direction_1[0, :, 21].cpu().detach().numpy(), label='index 21')
        # plt.plot(direction_1[0, :, 22].cpu().detach().numpy(), label='index 22')
        # plt.plot(direction_1[0, :, 23].cpu().detach().numpy(), label='index 23')
        # plt.plot(direction_1[0, :, 24].cpu().detach().numpy(), label='index 24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.grid(linewidth = 1.5)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()
        # plt.subplot(122)
        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_2[0, :, 0].cpu().detach().numpy(), label='(0,0),0')
        # plt.plot(direction_2[0, :, 1].cpu().detach().numpy(), label='(1,0),5')
        # plt.plot(direction_2[0, :, 2].cpu().detach().numpy(), label='(2,0),10')
        # plt.plot(direction_2[0, :, 3].cpu().detach().numpy(), label='(3,0),15')
        # plt.plot(direction_2[0, :, 4].cpu().detach().numpy(), label='(4,0),20')
        # plt.plot(direction_2[0, :, 5].cpu().detach().numpy(), label='(4,1),21')
        # plt.plot(direction_2[0, :, 6].cpu().detach().numpy(), label='(3,1),16')
        # plt.plot(direction_2[0, :, 7].cpu().detach().numpy(), label='(2,1),11')
        # plt.plot(direction_2[0, :, 8].cpu().detach().numpy(), label='(1,1),6')
        # plt.plot(direction_2[0, :, 9].cpu().detach().numpy(), label='(0,1),1')
        # plt.plot(direction_2[0, :, 10].cpu().detach().numpy(), label='(0,2),2')
        # plt.plot(direction_2[0, :, 11].cpu().detach().numpy(), label='(1,2),7')
        # plt.plot(direction_2[0, :, 12].cpu().detach().numpy(), label='(2,2), center', linewidth=3, linestyle='-.')
        # plt.plot(direction_2[0, :, 13].cpu().detach().numpy(), label='(3,2),17')
        # plt.plot(direction_2[0, :, 14].cpu().detach().numpy(), label='(4,2),22')
        # plt.plot(direction_2[0, :, 15].cpu().detach().numpy(), label='(4,3),23')
        # plt.plot(direction_2[0, :, 16].cpu().detach().numpy(), label='(3,3),18')
        # plt.plot(direction_2[0, :, 17].cpu().detach().numpy(), label='(2,3),13')
        # plt.plot(direction_2[0, :, 18].cpu().detach().numpy(), label='(1,3),8')
        # plt.plot(direction_2[0, :, 19].cpu().detach().numpy(), label='(0,3),3', linewidth=5)
        # plt.plot(direction_2[0, :, 20].cpu().detach().numpy(), label='(0,4),4', linewidth=5)
        # plt.plot(direction_2[0, :, 21].cpu().detach().numpy(), label='(1,4),9', linewidth=5)
        # plt.plot(direction_2[0, :, 22].cpu().detach().numpy(), label='(2,4),14')
        # plt.plot(direction_2[0, :, 23].cpu().detach().numpy(), label='(3,4),19')
        # plt.plot(direction_2[0, :, 24].cpu().detach().numpy(), label='(4,4),24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()

        # # plt.subplot(332)
        # plt.imshow(direction_1[0, :, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-1 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(333)
        # plt.imshow(direction_2[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-2 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(334)
        # plt.imshow(direction_3[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-3 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(335)
        # plt.imshow(direction_4[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-4 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(336)
        # plt.imshow(direction_5[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-5 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(337)
        # plt.imshow(direction_6[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-6 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(338)
        # plt.imshow(direction_7[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()ticks(fontsize=20)
        # plt.title('Direction-7 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(339)
        # plt.imshow(direction_8[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-8 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()

        # 换成输入顺序
        x8r = direction_8.permute(2, 0, 1)
        x7r = direction_7.permute(2, 0, 1)
        x6r = direction_6.permute(2, 0, 1)
        x5r = direction_5.permute(2, 0, 1)
        x4r = direction_4.permute(2, 0, 1)
        x3r = direction_3.permute(2, 0, 1)
        x2r = direction_2.permute(2, 0, 1)
        x1r = direction_1.permute(2, 0, 1)
        # print('d5.shape', x5r.shape)
        # plt.subplot(3, 4, 9)
        # plt.plot(direction_1[0, :, :].cpu().detach().numpy())

        'soft mask with multiscanning'
        # def softweight(x):
        #     x_weight = rearrange(x, 'x b c -> b x c')
        #     x_dist = torch.cdist(x_weight, x_weight, p=2)
        #     mean_x_dist = torch.mean(x_dist)
        #     x_weight_1 = torch.exp(-(x_dist ** 2) / 2 * (mean_x_dist ** 2))
        #     # g = sns.heatmap(x_weight_1[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        #     # g.set_title('')
        #     # plt.show()
        #     return x_weight_1
        # plt.subplot(241)
        # g1 = sns.heatmap(softweight(x1r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # # cbar = g1.collections[0].colorbar
        # # cbar.ax.tick_params(labelsize=20)
        # plt.subplot(242)
        # g2 = sns.heatmap(softweight(x2r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(243)
        # g3 = sns.heatmap(softweight(x3r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True, )
        # plt.subplot(244)
        # g4 = sns.heatmap(softweight(x4r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.subplot(245)
        # g5 = sns.heatmap(softweight(x5r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True, )
        # plt.subplot(246)
        # g6 = sns.heatmap(softweight(x6r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.subplot(247)
        # g7 = sns.heatmap(softweight(x7r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.subplot(248)
        # g8 = sns.heatmap(softweight(x8r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.show()

        # print('x1r',x1r.shape)
        '-----------------------------------------------------------------------------------------------------'
        h0_x1r = torch.zeros(1, x1r.size(1), 64).to(device='cuda')
        c0_x1r = torch.zeros(1, x1r.size(1), 64).to(device="cuda")
        x1r = self.lstm_2_1(x1r)

        # print('hidden', x1r_hidden.shape)
        # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)
        x1r_laststep = x1r[-1]
        x1r_laststep = self.relu(x1r_laststep)
        x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)
        # print('x1r last',x1r_laststep_2.shape)

        # 'calzulate RNN attention, direction 1'
        # allpixels1 = x1r[:,1,:]
        # allpixels1 = self.linear(allpixels1)
        # print('allpixels1', allpixels1.shape)
        # pairdistance = nn.PairwiseDistance(p=2)
        # x1r_laststep_2 = self.linear(x1r_laststep_2)
        # output1 = pairdistance(allpixels1,x1r_laststep_2)
        # output1 = self.softmax(output1)
        # output1 = output1.unsqueeze(0)
        #
        # output1_1 = torch.matmul(allpixels1,x1r_laststep_2)
        # output1_1 =self.softmax(output1_1)
        # output1_1 = output1_1.unsqueeze(0)
        # # print('output12',output12)
        # # plt.plot(output1_1.cpu().detach().numpy(), linewidth = 2, marker = 'o')
        # # plt.show()
        # # a1 = sns.heatmap(data=output1.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction1')
        # # plt.show()
        '----------------------------------------------------------------------------------------------------'
        h0_x2r = torch.zeros(1, x2r.size(1), 64).to(device='cuda')
        c0_x2r = torch.zeros(1, x2r.size(1), 64).to(device="cuda")
        x2r = self.lstm_2_2(x2r)
        # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        x2r_laststep = x2r[-1]
        # x2r_laststep_2 = x2r[-1, 1, :]
        x2r_laststep = self.relu(x2r_laststep)
        x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)

        # 'calzulate RNN attention, direction 2'
        # allpixels2 = x2r[:,1,:]
        # allpixels2 = self.linear(allpixels2)
        # x2r_laststep_2 = self.linear(x2r_laststep_2)
        # output2 = pairdistance(allpixels2,x2r_laststep_2)
        # output2 = self.softmax(output2)
        # output2 = output2.unsqueeze(0)
        #
        # output2_2 = torch.matmul(allpixels2,x2r_laststep_2)
        # output2_2 =self.softmax(output2_2)
        # output2_2 = output2_2.unsqueeze(0)
        # # plt.plot(output2[0,:].cpu().detach().numpy(), linewidth =1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output2.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction2')
        # # plt.show()
        '----------------------------------------------------------------------------------------------------------'

        h0_x3r = torch.zeros(1, x3r.size(1), 64).to(device='cuda')
        c0_x3r = torch.zeros(1, x3r.size(1), 64).to(device="cuda")
        x3r = self.lstm_2_3(x3r)
        # x3r = self.gru_2_3(x3r)[0]
        x3r_laststep = x3r[-1]
        # x3r_laststep_2 = x3r[-1, 1, :]
        x3r_laststep = self.relu(x3r_laststep)
        x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)

        # 'calzulate RNN attention, direction 3'
        # allpixels3 = x3r[:, 1, :]
        # allpixels3 =self.linear(allpixels3)
        # x3r_laststep_2 = self.linear(x3r_laststep_2)
        # output3 = pairdistance(allpixels3, x3r_laststep_2)
        # output3 = self.softmax(output3)
        # output3 = output3.unsqueeze(0)
        # print('output3', output3)
        #
        # output3_3 = torch.matmul(allpixels3,x3r_laststep_2)
        # output3_3 =self.softmax(output3_3)
        # output3_3 = output3_3.unsqueeze(0)
        # # plt.plot(output3[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output3.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction3')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x4r = torch.zeros(1, x4r.size(1), 64).to(device='cuda')
        c0_x4r = torch.zeros(1, x4r.size(1), 64).to(device="cuda")
        x4r = self.lstm_2_4(x4r)
        # x4r = self.gru_2_4(x4r)[0]
        x4r_laststep = x4r[-1]
        # x4r_laststep_2 = x4r[-1, 1, :]
        x4r_laststep = self.relu(x4r_laststep)
        x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)

        # 'calzulate RNN attention, direction 4'
        # allpixels4 = x4r[:, 1, :]
        # allpixels4 = self.linear(allpixels4)
        # x4r_laststep_2 = self.linear(x4r_laststep_2)
        # output4 = pairdistance(allpixels4, x4r_laststep_2)
        # output4 = self.softmax(output4)
        # output4 = output4.unsqueeze(0)
        #
        # output4_4 = torch.matmul(allpixels4,x4r_laststep_2)
        # output4_4 =self.softmax(output4_4)
        # output4_4 = output4_4.unsqueeze(0)
        # # plt.plot(output4[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output4.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction4')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x5r = torch.zeros(1, x5r.size(1), 64).to(device='cuda')
        c0_x5r = torch.zeros(1, x5r.size(1), 64).to(device="cuda")
        x5r = self.lstm_2_5(x5r)
        # x5r = self.gru_2_5(x5r)[0]
        x5r_laststep = x5r[-1]
        # x5r_laststep_2 = x5r[-1, 1, :]
        x5r_laststep = self.relu(x5r_laststep)
        x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)

        # 'calzulate RNN attention, direction 5'
        # allpixels5 = x5r[:, 1, :]
        # allpixels5 = self.linear(allpixels5)
        # x5r_laststep_2 = self.linear(x5r_laststep_2)
        # output5 = pairdistance(allpixels5, x5r_laststep_2)
        # output5 = self.softmax(output5)
        # output5 = output5.unsqueeze(0)
        #
        # output5_5 = torch.matmul(allpixels5,x5r_laststep_2)
        # output5_5 =self.softmax(output5_5)
        # output5_5 = output5_5.unsqueeze(0)
        # # plt.plot(output5[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output5.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction5')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x6r = torch.zeros(1, x6r.size(1), 64).to(device='cuda')
        c0_x6r = torch.zeros(1, x6r.size(1), 64).to(device="cuda")
        x6r = self.lstm_2_6(x6r)
        # x6r = self.gru_2_6(x6r)[0]
        x6r_laststep = x6r[-1]
        # x6r_laststep_2 = x6r[-1, 1, :]
        x6r_laststep = self.relu(x6r_laststep)
        x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)

        # 'calzulate RNN attention, direction 6'
        # allpixels6 = x6r[:, 1, :]
        # allpixels6 = self.linear(allpixels6)
        # x6r_laststep_2 = self.linear(x6r_laststep_2)
        # output6 = pairdistance(allpixels6, x6r_laststep_2)
        # output6 = self.softmax(output6)
        # output6 = output6.unsqueeze(0)
        #
        # output6_6 = torch.matmul(allpixels6,x6r_laststep_2)
        # output6_6 =self.softmax(output6_6)
        # output6_6 = output6_6.unsqueeze(0)
        # # plt.plot(output6[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output6.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction6')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x7r = torch.zeros(1, x7r.size(1), 64).to(device='cuda')
        c0_x7r = torch.zeros(1, x7r.size(1), 64).to(device="cuda")
        x7r = self.lstm_2_7(x7r)
        # x7r = self.gru_2_7(x7r)[0]
        x7r_laststep = x7r[-1]
        # x7r_laststep_2 = x7r[-1, 1, :]
        x7r_laststep = self.relu(x7r_laststep)
        x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        #
        # 'calzulate RNN attention, direction 7'
        # allpixels7 = x7r[:, 1, :]
        # allpixels7 = self.linear(allpixels7)
        # x7r_laststep_2 = self.linear(x7r_laststep_2)
        # output7 = pairdistance(allpixels7, x7r_laststep_2)
        # output7 = self.softmax(output7)
        # output7 = output7.unsqueeze(0)
        #
        # output7_7 = torch.matmul(allpixels7,x7r_laststep_2)
        # output7_7 =self.softmax(output7_7)
        # output7_7 = output7_7.unsqueeze(0)
        # # plt.plot(output7[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output7.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction7')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x8r = torch.zeros(1, x8r.size(1), 64).to(device='cuda')
        c0_x8r = torch.zeros(1, x8r.size(1), 64).to(device="cuda")
        x8r = self.lstm_2_8(x8r)
        # x8r = self.gru_2_8(x8r)[0]
        x8r_laststep = x8r[-1]
        # x8r_laststep_2 = x8r[-1, 1, :]
        x8r_laststep = self.relu(x8r_laststep)
        x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        # print('x8r_last',x8r_laststep.shape)

        # 'calzulate RNN attention, direction 8'
        # allpixels8 = x8r[:, 1, :]
        # allpixels8 = self.linear(allpixels8)
        # x8r_laststep_2 = self.linear(x8r_laststep_2)
        # output8 = pairdistance(allpixels8, x8r_laststep_2)
        # output8 = self.softmax(output8)
        # output8 = output8.unsqueeze(0)
        #
        # output8_8 = torch.matmul(allpixels8,x8r_laststep_2)
        # output8_8 =self.softmax(output8_8)
        # output8_8 = output8_8.unsqueeze(0)
        # # plt.plot(output8[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output8.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction8')
        # # plt.show()
        '----show attetntion function------------------------------------------------------'
        # def showattention(inputseq):
        #     allpixel = inputseq[:, 1, :]
        #     linear1 = nn.Linear(allpixel.size(1),allpixel.size(1)).to( device='cuda')
        #     allpixel = linear1(allpixel)
        #
        #     # centralstep = allpixel[12,:]
        #     laststep = inputseq[-1, 1, :]
        #     laststep = linear1(laststep)
        #
        #     output = torch.matmul(allpixel, laststep.transpose(0,-1))
        #
        #     pairdis = nn.PairwiseDistance()
        #     cos = nn.CosineSimilarity(dim=-1)
        #
        #     output_pair = pairdis(allpixel,laststep) * -1
        #     # output_pair = cos(allpixel, laststep)
        #
        #     softmax = nn.Softmax()
        #     output = softmax(output)
        #     output_pair = softmax(output_pair)
        #     output = output.unsqueeze(0)
        #     output_pair = output_pair.unsqueeze(0)
        #     print('cos',output_pair.shape)
        #     return output,output_pair
        '------------------------------------------------------------------------------------'
        # output1_1,output1_1_cos = showattention(x1r)
        # # sns.lineplot(data=output1_1_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output2_2, output2_2_cos = showattention(x2r)
        # # sns.lineplot(data=output2_2_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output3_3,output3_3_cos = showattention(x3r)
        # # sns.lineplot(data=output3_3_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output4_4,output4_4_cos = showattention(x4r)
        # # sns.lineplot(data=output4_4_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output5_5,output5_5_cos = showattention(x5r)
        # # sns.lineplot(data=output5_5_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output6_6,output6_6_cos = showattention(x6r)
        # # sns.lineplot(data=output6_6_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output7_7,output7_7_cos = showattention(x7r)
        # # sns.lineplot(data=output7_7_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output8_8,output8_8_cos = showattention(x8r)
        # # sns.lineplot(data=output8_8_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'
        # outputall = torch.cat([output1_1_cos, output2_2_cos, output3_3_cos, output4_4_cos, output5_5_cos, output6_6_cos, output7_7_cos, output8_8_cos],dim=0)
        # sns.lineplot(data=outputall.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        # all = sns.heatmap(data=outputall.cpu().detach().numpy(), cmap="mako", linewidths=0.05, square=True)
        # all.set_title('all')
        # plt.show()

        '第一种合并'
        # # x = x1r_laststep + x2r_laststep + x3r_laststep + x4r_laststep + x5r_laststep + x6r_laststep + x7r_laststep + x8r_laststep
        # # x = x.squeeze(0)
        # x = x1r + x2r + x3r+ x4r + x5r + x6r + x7r + x8r
        # print('x.shape',x.shape)
        # x = x.permute(1, 2, 0).contiguous()
        # # x = x.permute(0, 1).contiguous()
        # x = x.view(x.size(0), -1)
        # x = self.lstm_bn_1_2(x)
        # x = self.prelu(x)
        # # x = self.dropout(x)
        # x = self.lstm_fc_1_2(x)

        # '第二种合并'
        # x = torch.cat([x1r_laststep, x2r_laststep, x3r_laststep, x4r_laststep, x5r_laststep, x6r_laststep, x7r_laststep,
        #                x8r_laststep], dim=2)
        # # x = torch.cat([x1r,x2r,x3r,x4r,x5r,x6r,x7r,x8r],dim=2)
        # print('x.shape',x.shape)
        # x = x.squeeze(0)
        # x = x.permute(0, 1).contiguous()
        # # x = x.permute(1, 2, 0).contiguous()
        # x = x.view(x.size(0), -1)
        # x = self.lstm_bn_2(x)
        # # x = self.lstm_bn_2_2(x)
        # x = self.prelu(x)
        # x = self.dropout(x)
        # x = self.lstm_fc_2(x)
        # # x = self.lstm_fc_2_2(x)

        "第三种合并"
        x_strategy_1 = torch.cat(
            [x1r_laststep, x3r_laststep, x8r_laststep, x4r_laststep, x5r_laststep, x6r_laststep, x7r_laststep,
             x2r_laststep], dim=0)
        x_strategy_1 = rearrange(x_strategy_1, 'n b d -> b n d')
        # x_strategy_1 = self.scan_order(x_strategy_1)[0]
        # print('x_strategy_1', x_strategy_1.shape) #(8 , batch, 64)
        # x_strategy_1 = x_strategy_1.permute(1, 0, 2).contiguous()#(100,64,8)
        # h0_last = torch.zeros(1, x_strategy_1.size(1), 64).to(device='cuda')
        # c0_last = torch.zeros(1, x_strategy_1.size(1), 64).to(device="cuda")
        x_strategy_1 = self.lstm_stra_1(x_strategy_1)
        # print('output2', x_strategy_1.shape)
        # x_strategy_1 = self.gru_4(x_strategy_1)[0]
        x_strategy_1_laststep = x_strategy_1
        # print('output3', x_strategy_1_laststep.shape)
        # x_strategy_1_laststep_2 = x_strategy_1[-1, 1, :]
        # x_strategy_1_laststep = x_strategy_1.permute(1, 2, 0).contiguous()
        # print('x_strategy_1_laststep',x_strategy_1_laststep.shape)
        # np.save('x_strategy_1_laststep', x_strategy_1_laststep.cpu().detach().numpy(), allow_pickle=True)
        '------------------------------------------'
        'calzulate RNN attention for 8 directions'

        '-------------------------------------------------------------------------------------'
        # x_strategy_1_laststep = x_strategy_1_laststep.permute(0, 1).contiguous()
        # x_strategy_1_laststep = x_strategy_1_laststep.view(x_strategy_1_laststep.size(0), -1)
        # x_strategy_1_laststep = self.gru_bn_4(x_strategy_1_laststep)
        x_strategy_1_laststep = x_strategy_1_laststep.permute(0, 2, 1).contiguous()
        x_strategy_1_laststep = x_strategy_1_laststep.view(x_strategy_1_laststep.size(0), -1)
        x_strategy_1_laststep = self.gru_bn_laststep(x_strategy_1_laststep)
        x_strategy_1_laststep = self.prelu(x_strategy_1_laststep)
        x_strategy_1_laststep = self.dropout(x_strategy_1_laststep)
        # x_strategy_1_laststep = self.fc_4(x_strategy_1_laststep)
        x_strategy_1_laststep_cls = self.fc_laststep(x_strategy_1_laststep)
        # x_strategy_1_laststep_reg = self.reg(x_strategy_1_laststep)
        # var2 = torch.var(x_strategy_1_laststep)
        # print('var2:', var2)

        x_cls = x_strategy_1_laststep_cls
        # print('output4', x_cls.shape)
        # x_reg = x_strategy_1_laststep_reg
        # 下面改变输入值，确定使用哪个方向

        # x1r = x1r.permute(1, 2, 0).contiguous()
        # x2r = x2r.permute(1, 2, 0).contiguous()
        # x3r = x3r.permute(1, 2, 0).contiguous()
        # x4r = x4r.permute(1, 2, 0).contiguous()
        # x5r = x5r.permute(1, 2, 0).contiguous()
        # x6r = x6r.permute(1, 2, 0).contiguous()
        # x7r = x7r.permute(1, 2, 0).contiguous()
        # x8r = x8r.permute(1, 2, 0).contiguous()
        # x_strategy_1 = x_strategy_1.permute(1,2,0).contiguous()

        # x1r = x1r.view(x1r.size(0), -1)
        # x2r = x2r.view(x2r.size(0), -1)
        # x3r = x3r.view(x3r.size(0), -1)
        # x4r = x4r.view(x4r.size(0), -1)
        # x5r = x5r.view(x5r.size(0), -1)
        # x6r = x6r.view(x6r.size(0), -1)
        # x7r = x7r.view(x7r.size(0), -1)
        # x8r = x8r.view(x8r.size(0), -1)
        # x_strategy_1 = x_strategy_1.view(x_strategy_1.size(0),-1)

        # x1r = self.gru_bn_3(x1r)
        # x2r = self.gru_bn_3(x2r)
        # x3r = self.gru_bn_3(x3r)
        # x4r = self.gru_bn_3(x4r)
        # x5r = self.gru_bn_3(x5r)
        # x6r = self.gru_bn_3(x6r)
        # x7r = self.gru_bn_3(x7r)
        # x8r = self.gru_bn_3(x8r)
        # x_strategy_1 = self.gru_bn_4(x_strategy_1)

        # x1r = self.tanh(x1r)
        # x2r = self.tanh(x2r)
        # x3r = self.tanh(x3r)
        # x4r = self.tanh(x4r)
        # x5r = self.tanh(x5r)
        # x6r = self.tanh(x6r)
        # x7r = self.tanh(x7r)
        # x8r = self.tanh(x8r)
        #
        # x1r = self.dropout(x1r)
        #
        # x2r = self.dropout(x2r)
        #
        # x3r = self.dropout(x3r)
        #
        #
        # x4r = self.dropout(x4r)
        #
        #
        # x5r = self.dropout(x5r)
        #
        #
        # x6r = self.dropout(x6r)
        #
        #
        # x7r = self.dropout(x7r)
        #
        #
        # x8r = self.dropout(x8r)
        # x_strategy_1 = self.dropout(x_strategy_1)
        # x_strategy_1_laststep = self.dropout(x_strategy_1_laststep)

        # plt.subplot(3, 3, 1).set_title('Spectral signatures in a patch')
        # plt.xlabel('Band Numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 15}, fontweight='bold')
        # plt.plot(direction_1[0, :, :].cpu().detach().numpy())

        # x1r = self.fc_3(x1r)
        # plt.subplot(3, 3, 2)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x1r[0, :].cpu().detach().numpy())

        # x2r = self.fc_3(x2r)
        # plt.subplot(3, 3, 3)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x2r[0, :].cpu().detach().numpy())

        # x3r = self.fc_3(x3r)
        # plt.subplot(3, 3, 4)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x3r[0, :].cpu().detach().numpy())

        # x4r = self.fc_3(x4r)
        # plt.subplot(3, 3, 5)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x4r[0, :].cpu().detach().numpy())

        # x5r = self.fc_3(x5r)
        # plt.subplot(3, 3, 6)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x5r[0, :].cpu().detach().numpy())

        # x6r = self.fc_3(x6r)
        # plt.subplot(3, 3, 7)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x6r[0, :].cpu().detach().numpy())

        # x7r = self.fc_3(x7r)
        # plt.subplot(3, 3, 8)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x7r[0, :].cpu().detach().numpy())

        # x8r = self.fc_3(x8r)
        # plt.subplot(3, 3, 9)
        # plt.xlabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x8r[0, :].cpu().detach().numpy())
        # plt.show()

        # x_strategy_1 = self.fc_4(x_strategy_1)'Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x8r[0, :].cpu().detach().numpy())
        # plt.show()

        # x_strategy_1 = self.fc_4(x_strategy_1)

        return x_cls


class Order_Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Order_Attention, self).__init__()
        self.w_omega = nn.Parameter(torch.randn(hidden_size, attention_size))  # [hidden_size, attention_size]
        self.b_omega = nn.Parameter(torch.randn(attention_size))  # [attention_size]
        self.u_omega = nn.Parameter(torch.randn(attention_size))  # [attention_size]

    def forward(self, inputs):
        # inputs: [seq_len, batch_size, hidden_size]
        inputs = inputs.permute(1, 0, 2)  # inputs: [batch_size, seq_len, hidden_size]
        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega)  # v: [batch_size, seq_len, attention_size]
        vu = torch.matmul(v, self.u_omega)  # vu: [batch_size, seq_len]
        alphas = F.softmax(vu, dim=1)  # alphas: [batch_size, seq_len]
        output = inputs * alphas.unsqueeze(-1)  # output: [batch_size, STEP, hidden_size]
        # print('output', output.shape)
        return output, alphas  # output: [batch_size, hidden_size], alphas: [batch_size, seq_len]


class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_size):
        super(ARNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=False)  # output: [seq_len, batch_size, 2*hidden_size]
        self.attention = Order_Attention(hidden_size, attention_size)
        self.fc = nn.Linear(hidden_size, output_size)  # output: [batch_size, output_size]

    def forward(self, x):
        # x: [seq_len, batch_size, input_size]
        outputs, _ = self.gru(x)  # outputs: [seq_len, batch_size, 2*hidden_size]
        a_output, alphas = self.attention(
            outputs)  # a_output: [batch_size, 2*hidden_size], alphas: [batch_size, seq_len]
        outputs = self.fc(a_output)  # outputs: [batch_size, output_size]
        return outputs  # outputs: [batch_size, output_size]
