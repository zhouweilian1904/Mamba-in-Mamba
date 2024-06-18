import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class SpectralAttention(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data)
            init.uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, embed_fea = 64, num_layer=1):
        super(SpectralAttention, self).__init__()
        self.aux_loss_weight = 1
        self.input_channels = input_channels
        self.gru = nn.GRU(1, embed_fea, num_layer,
                            bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_1 = nn.GRU(1,embed_fea, num_layer, bidirectional= False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_2 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_3 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_4 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_5 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_6 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_7 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_8 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_9 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_10 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        # self.gru2 = nn.GRU(1,1,1)
        self.gru3 = nn.GRU(embed_fea,embed_fea,embed_fea, bidirectional= False)#之前是（64,64，1）  pavia——100用100的feature hidden。 indianpine用200如果用双向改成100
        # self.gru4 = nn.GRU(1,200,1)#之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_bn = nn.BatchNorm1d(embed_fea )#之前是（64*64）根据 记得根据数据集更改input——channel pavia——100用100的feature hidden。 indianpine用200
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(embed_fea , n_classes) #之前是（64*64,）记得根据数据集更改input——channel pavia——100用100的feature hidden。 indianpine用200
        self.regressor = nn.Linear(embed_fea,input_channels)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        # self.model2 = HuEtAl(input_channels=input_channels,n_classes=n_classes)
        self.aux_loss_weight = 1

    def forward(self, x):
        # print('1',x.shape)
        # pre2 = self.model2(x)
        # x = x.squeeze()
        # print('2',x.shape)
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        # print('3',x.shape)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        # plt.subplot(4,1,1)
        # plt.plot(x[:, 1, :].cpu().detach().numpy(),linewidth='2.5')
        # print('4',x.shape)
        x1_10,x11_20,x21_30,x31_40,x41_50,x51_60,x61_70,x71_80,x81_90,x91_100 = x.chunk(10,0)
        # plt.subplot(3, 3, 1)
        # plt.plot(x1_10[:,1, :].cpu().detach().numpy())
        # plt.subplot(3, 3, 2)
        # plt.plot(x11_20[:,1,:].cpu().detach().numpy())
        # plt.subplot(3, 3, 3)
        # plt.plot(x21_30[:,1, :].cpu().detach().numpy())
        x1_10 = self.gru_1(x1_10)[0]
        # print('x1_10',x1_10.shape)
        x1_10_l = x1_10[-1] #64
        x1_10_l = torch.unsqueeze(x1_10_l, 0)
        # plt.subplot(3,3,4)
        # plt.plot(x1_10[:, 1, :].cpu().detach().numpy())
        # plt.plot(x1_10_l[1,:].cpu().detach().numpy())
        # plt.subplot(3,3,7)
        # plt.plot(x1_10_l.cpu().detach().numpy())
        x11_20 = self.gru_2(x11_20)[0]
        x11_20_l = x11_20[-1]
        x11_20_l = torch.unsqueeze(x11_20_l, 0)
        # print('x_11_20_l', x11_20_l.shape) #(1, batch, 64)
        # plt.subplot(3,3,5)
        # plt.plot(x11_20[:, 1, :].cpu().detach().numpy())
        x21_30 = self.gru_3(x21_30)[0]
        x21_30_l = x21_30[-1]
        x21_30_l = torch.unsqueeze(x21_30_l, 0)
        # plt.subplot(3,3,6)
        # plt.plot(x21_30[:, 1, :].cpu().detach().numpy())
        x31_40 = self.gru_4(x31_40)[0]
        x31_40_l = x31_40[-1]
        x31_40_l = torch.unsqueeze(x31_40_l,0)

        x41_50 = self.gru_5(x41_50)[0]
        x41_50_l = x41_50[-1]
        x41_50_l = torch.unsqueeze(x41_50_l, 0)

        x51_60 = self.gru_6(x51_60)[0]
        x51_60_l = x51_60[-1]
        x51_60_l = torch.unsqueeze(x51_60_l, 0)

        x61_70 = self.gru_7(x61_70)[0]
        x61_70_l = x61_70[-1]
        x61_70_l = torch.unsqueeze(x61_70_l, 0)

        x71_80 = self.gru_8(x71_80)[0]
        x71_80_l = x71_80[-1]
        x71_80_l = torch.unsqueeze(x71_80_l, 0)

        x81_90 = self.gru_9(x81_90)[0]
        x81_90_l = x81_90[-1]
        x81_90_l = torch.unsqueeze(x81_90_l,0)

        x91_100 = self.gru_10(x91_100)[0]
        x91_100_l = x91_100[-1]
        x91_100_l = torch.unsqueeze(x91_100_l,0)  #size 1,batch,feature
        # print('x_91_100_l',x91_100_l.shape)
        # x91_100_l = x91_100_l.expand(x91_100_l.shape[0],10) 这个可以把seq扩张成matrix

        x_cat = torch.cat([x1_10_l,x11_20_l,x21_30_l,x31_40_l,x41_50_l,x51_60_l,x61_70_l,x71_80_l,x81_90_l,x91_100_l],dim=0)  #size 10,batch,feature
        # print('x_cat', x_cat.shape) # 10, batch, 64
        x_cat = self.gru3(x_cat)[0]
        # print('x_cat', x_cat.shape)# 10, batch, 64
        x_cat_l = torch.avg_pool1d(rearrange(x_cat,'l n c -> n c l'),kernel_size=10)
        # x_cat_l = x_cat[-1] #size  (batch,feature=64)
        # print('x_cat_l[-1]',x_cat_l.shape) #batch,64
        x_cat_l = self.relu(x_cat_l)
        # plt.subplot(4, 1, 3)
        # plt.plot(x_cat_l[1, :].cpu().detach().numpy(),linewidth='2.5')
        x = self.gru(x)[0]
        # print('x:',x.shape) # l n c
        x_l = torch.avg_pool1d(rearrange(x, 'l n c -> n c l'), kernel_size=self.input_channels)
        # x_l = x[-1] #size batch,feature=64
        # plt.subplot(4, 1, 2)
        # plt.plot(x_l[1, :].cpu().detach().numpy(),linewidth='2.5')
        # print('x_l[-1]',x_l.shape) # 1 100 64
        x_new = x_l * x_cat_l  #改成+号试试
        # print('x_new',x_new.shape)
        # plt.subplot(4, 1, 4)
        # plt.plot(x_new[1, :].cpu().detach().numpy(),linewidth='5')
        # x_new = torch.unsqueeze(x_new,0)
        # print('x_new',x_new.shape) # 1 , batch, 64
        # x_new = x_new.permute(2,1,0) #feature 变为 seq
        # print('x_new',x_new.shape) #size: seq,batch,1
        x = x_new
        # plt.subplot(4,1,2)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # x_cat = self.gru2(x_cat)[0]
        # x_cat = self.tanh(x_cat)
        # plt.subplot(4,1,3)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # plt.subplot(3, 3, 8)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # print('x1_10', x1_10.shape)
        # print('x91_100', x91_100.shape)
        # plt.subplot(142)
        # plt.plot(x_cat[:,1,:].cpu().detach().numpy())
        # x_cat = self.gru4(x_cat)[0]
        # plt.subplot(143)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # x_cat_1 = x_cat[-1]
        # # x_cat_1 = self.relu(x_cat_1)
        # plt.subplot(144)
        # plt.plot(x_cat_1.cpu().detach().numpy())
        # x = x + x_cat
        # x = x * x_cat
        # plt.subplot(4,1,4)
        # plt.plot(x[:,1,:].cpu().detach().numpy())
        # plt.subplot(3, 3, 9)
        # plt.plot(x[:, 1, :].cpu().detach().numpy())
        # plt.imshow(x[:,1,:].cpu().detach().numpy())
        # x = self.gru(x_new)[0]
        # plt.subplot(144)
        # plt.plot(x[:, 1, :].cpu().detach().numpy())
        # plt.show()
        # x = self.gru3(x)[0]
        # print('5',x.shape)
        # x is in C, N, 64, we permute back
        # x = x.permute(1, 2, 0).contiguous()
        # print('6',x.shape)
        x = x.view(x.size(0), -1)
        # print('7',x.shape)
        # x = nn.BatchNorm1d(x.shape[1])
        x = self.gru_bn(x)
        # x = self.tanh(x)
        x = self.relu(x)
        x_rec = self.regressor(x)
        # x = nn.Linear(x.shape[1],10)
        # x = self.dropout(x)
        x_class = self.fc(x)
        # plt.grid(linewidth = 0.5, color = 'black' )
        # plt.title('Visualiazations in the block', fontdict={'size':40})
        # plt.legend(['x','ff','fx','fnew'], prop={'size':40}, fontsize = 'large')
        # plt.xlabel('Feature size', fontdict={'size':40}, fontweight = 'bold')
        # plt.ylabel('Feature value',fontdict={'size':40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # plt.show()
        # print('after fc', x.shape)
        return x_class, x_rec