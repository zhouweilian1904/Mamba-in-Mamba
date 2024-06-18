import torch.nn as nn
import torch
from torch.nn import init
import numpy as np
from einops import rearrange, reduce, repeat

class ICPR2022(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM, nn.TransformerEncoderLayer, nn.TransformerEncoder)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=9, embed_dim = 64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(ICPR2022, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        'do not use self.n_classes = n_classes'
        self.pos_embedding_81_plus_1 = nn.Parameter(torch.randn(1, 81 + 1, self.input_channels ))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding_49_plus_1 = nn.Parameter(torch.randn(1, 49+1, self.input_channels ))
        self.cls_token_49_plus_1 = nn.Parameter(torch.randn(1, 1, self.input_channels))
        self.pos_embedding_25_plus_1 = nn.Parameter(torch.randn(1, 25+1, self.input_channels ))
        self.cls_token_25_plus_1 = nn.Parameter(torch.randn(1, 1, self.input_channels))
        self.pos_embedding_9_plus_1 = nn.Parameter(torch.randn(1, 9+1, self.input_channels ))
        self.cls_token_9_plus_1 = nn.Parameter(torch.randn(1, 1, self.input_channels))
        self.cls_token_FLC = nn.Parameter(torch.randn(1, 1, embed_dim * 2))

        self.alpha = nn.Parameter(torch.randn(1,1))
        self.beta = nn.Parameter(torch.randn(1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1))

        self.conv_11to9 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels ,kernel_size=3)
        self.encoder_layer_81_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64, activation='gelu')
        self.transformer_encoder_81_plus_1 = nn.TransformerEncoder(self.encoder_layer_81_plus_1, num_layers=1,norm=None)


        self.conv_9to7 = nn.Conv2d(in_channels=self.input_channels , out_channels=self.input_channels , kernel_size=3)
        self.encoder_layer_49_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64,activation='gelu')
        self.transformer_encoder_49_plus_1 = nn.TransformerEncoder(self.encoder_layer_49_plus_1, num_layers=1, norm=None)
        self.deconv_1to7 = nn.ConvTranspose2d(in_channels=input_channels,out_channels=input_channels,kernel_size=6,stride=1)

        self.conv_7to5 = nn.Conv2d(in_channels=self.input_channels ,out_channels=self.input_channels ,kernel_size=3)
        self.encoder_layer_25_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64,activation='gelu')
        self.transformer_encoder_25_plus_1 = nn.TransformerEncoder(self.encoder_layer_25_plus_1, num_layers=1, norm=None)
        self.deconv_1to5 = nn.ConvTranspose2d(in_channels=input_channels,out_channels=input_channels,kernel_size=4,stride=1)

        self.conv_5to3 = nn.Conv2d(in_channels=self.input_channels , out_channels=self.input_channels , kernel_size=3)
        self.encoder_layer_9_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64,activation='gelu')
        self.transformer_encoder_9_plus_1 = nn.TransformerEncoder(self.encoder_layer_9_plus_1, num_layers=1, norm=None)
        self.deconv_1to3 = nn.ConvTranspose2d(in_channels=input_channels,out_channels=input_channels, kernel_size=2,stride=1)

        self.conv_3to1 = nn.Conv2d(in_channels=self.input_channels , out_channels=self.input_channels , kernel_size=3)

        self.adapooling = nn.AdaptiveAvgPool1d(1)
        self.adapooling2d = nn.AdaptiveAvgPool2d((1,1))
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64)*1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size)**2)
        self.lstm_bn_2 = nn.BatchNorm1d((64)*8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size**2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size**2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.transformer_bn_scheme1 = nn.BatchNorm1d(embed_dim)
        self.transformer_bn_scheme2 = nn.BatchNorm1d(embed_dim * 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(input_channels *3 , n_classes)
        self.bn = nn.BatchNorm1d(input_channels *3  )
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size**2), n_classes)
        self.lstm_fc_2 = nn.Linear(64*8,n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size**2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.denselayer = nn.Linear(input_channels, input_channels)
        self.denselayer1 = nn.Linear(input_channels * 2, input_channels * 2)
        self.denselayer2 = nn.Linear(input_channels * 3, input_channels * 3)
        self.denselayer3 = nn.Linear(input_channels * 4, input_channels * 4)
        self.denselayer4 = nn.Linear(input_channels * 5, input_channels * 5)
        self.denselayer_scheme1 = nn.Linear(embed_dim,embed_dim)
        self.denselayer_scheme2 = nn.Linear(embed_dim * 2, embed_dim * 2)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.fc_transformer_scheme1 = nn.Linear(embed_dim,n_classes)
        self.fc_transformer_scheme2 = nn.Linear(embed_dim * 2 , n_classes)
        self.linearembedding = nn.Linear(in_features=input_channels,out_features=embed_dim)
        self.softmax = nn.Softmax()
        self.norm = nn.LayerNorm(input_channels)
        self.cos = nn.CosineSimilarity(dim=1)
        self.distance = nn.PairwiseDistance(p=2)


    def forward(self, x): #初始是第1方向
        # print('x.shape1',x.shape)

        # x_11by11 = x.squeeze(1)
        # '11*11 --conv--> 9*9'
        # x_9by9 = self.conv_11to9(x_11by11)
        # x_9by9_flat = rearrange(x_9by9, 'b c h w -> b (h w) c')
        # b, n, c = x_9by9_flat.shape
        # x_9by9_flat += self.pos_embedding_81_plus_1[:, :(x_9by9_flat.size(1))]
        # print('x_9by9_flat_plus_cls', x_9by9_flat.shape)
        # x_9by9_flat_plus_cls = rearrange(x_9by9_flat, 'b n c -> n b c')
        # x_9by9_flat_plus_cls_output = self.transformer_encoder_81_plus_1(x_9by9_flat_plus_cls)
        # x_9by9_flat_plus_cls_output = self.denselayer(x_9by9_flat_plus_cls_output)
        # print('x_9by9_flat_plus_cls_output', x_9by9_flat_plus_cls_output.shape)
        # central_token = x_9by9_flat_plus_cls_output[40, :, :]
        # central_token = repeat(central_token, 'b c -> b () c')
        # central_token = repeat(central_token, 'b () c -> b c 9 9')
        # x_9by9 = central_token

        x_9by9 = x.squeeze(1)
        '9*9 --conv--> 7*7'
        x_7by7 = self.conv_9to7(x_9by9)
        x_7by7_base = x_7by7
        x_7by7_flat = rearrange(x_7by7, 'b c h w -> b (h w) c')
        #cls token + positional embeding
        b,n,c = x_7by7_flat.shape
        cls_tokens_49_plus_1 = repeat(self.cls_token_49_plus_1, '() n c -> b n c', b=b)
        x_7by7_flat_plus_cls = torch.cat((cls_tokens_49_plus_1, x_7by7_flat), dim=1)
        # print('x_7by7_flat_plus_cls',x_7by7_flat_plus_cls.shape)

        x_7by7_flat_plus_cls += self.pos_embedding_49_plus_1[:, :(x_7by7_flat_plus_cls.size(1) + 1)]
        # print('x_7by7_flat_plus_cls',x_7by7_flat_plus_cls.shape)
        # x_7by7_flat_plus_cls = self.dropout(x_7by7_flat_plus_cls)
        #trans setting

        # train trans
        x_7by7_flat_plus_cls = rearrange(x_7by7_flat, 'b n c -> n b c')
        x_7by7_flat_plus_cls_output = self.transformer_encoder_49_plus_1(x_7by7_flat_plus_cls)
        # plt.imshow(x_7by7_flat_plus_cls_output[:, 0, :].cpu().detach().numpy())
        x_7by7_flat_plus_cls_output = self.denselayer(x_7by7_flat_plus_cls_output)
        cls_token_7by7 = x_7by7_flat_plus_cls_output[0, :, :]
        # print('x_7by7_flat_plus_cls_output',x_7by7_flat_plus_cls_output.shape) #(49+1, b, channel)
        x_7by7_flat_plus_cls_output_pooling = rearrange((x_7by7_flat_plus_cls_output - x_7by7_flat_plus_cls_output[0,:,:]), 'l n c -> n c l ')
        # plt.show()
        # plt.plot(cls_token_7by7[0, :].cpu().detach().numpy(), color='orange')
        central_token_7by7 = x_7by7_flat_plus_cls_output[25,:,:]
        # plt.plot(central_token_7by7[0, :].cpu().detach().numpy(), color='red')
        pooling_token_7by7 = self.adapooling(x_7by7_flat_plus_cls_output_pooling)
        pooling_token_7by7 = reduce(pooling_token_7by7, 'b c 1 -> b c', reduction='mean')
        # plt.plot(pooling_token_7by7[0, :].cpu().detach().numpy(), color='blue')
        # plt.show()
        # print('pooling',pooling_token_7by7.shape)
        # print('cls:',cls_token_7by7.shape)
        # print('central:', central_token_7by7.shape)
        # print("cls_pooling 1:", self.cos(cls_token_7by7[:,:],pooling_token_7by7[:,:]), "cls_central:",self.cos(cls_token_7by7[:,:],central_token_7by7[:,:]), "pooling_central", self.cos(pooling_token_7by7[:,:],central_token_7by7[:,:]))

        #adaptive pooling
        # cls_token_7by7 = repeat(cls_token_7by7, 'b c -> b () c')
        # central_token_7by7 = repeat(central_token_7by7, 'b c -> b () c')
        # print('cls:', cls_token_49_plus_1_output.shape)
        # print('central:', central_token.shape)
        # cls_token_49_plus_1_output = self.adapooling(cls_token_49_plus_1_output)
        # central_token = self.adapooling(central_token)
        # print('cls:', cls_token_49_plus_1_output.shape)
        # print('central:', central_token_7by7.shape)
        # cls_token_49_plus_1_output = repeat(cls_token_49_plus_1_output, 'b () c -> b c 7 7') #先试试直接变成7by7
        # central_token = repeat(central_token, 'b () c -> b c 7 7')
        # central_token = self.deconv_1to7(central_token)
        # print('cls:', cls_token_49_plus_1_output.shape)
        # print('deconv_central:', central_token.shape)
        #重建成7by7的patch
        x_7by7 = x_7by7_base
        # x_7by7 = torch.cat((cls_token_49_plus_1_output,central_token), dim=1)
        # print('x_7by7',x_7by7.shape)

        '7*7 --conv--> 5*5'
        x_5by5 = self.conv_7to5(x_7by7)
        x_5by5_base = x_5by5
        x_5by5_flat = rearrange(x_5by5, 'b c h w -> b (h w) c')
        # cls token + positional embeding
        b, n, c = x_5by5_flat.shape
        cls_tokens_25_plus_1 = repeat(self.cls_token_25_plus_1, '() n c -> b n c', b=b)
        x_5by5_flat_plus_cls = torch.cat((cls_tokens_25_plus_1, x_5by5_flat), dim=1)
        # print('x_5by5_flat_plus_cls:', x_5by5_flat_plus_cls.shape)

        x_5by5_flat_plus_cls += self.pos_embedding_25_plus_1[:, :(x_5by5_flat_plus_cls.size(1) + 1)]
        # print('x_5by5_flat_plus_cls:', x_5by5_flat_plus_cls.shape)
        # x_5by5_flat_plus_cls = self.dropout(x_5by5_flat_plus_cls)
        # trans setting
        # train trans
        x_5by5_flat_plus_cls = rearrange(x_5by5_flat_plus_cls, 'b n c -> n b c')
        x_5by5_flat_plus_cls_output = self.transformer_encoder_25_plus_1(x_5by5_flat_plus_cls)
        x_5by5_flat_plus_cls_output = self.denselayer(x_5by5_flat_plus_cls_output)
        # print('x_5by5_flat_plus_cls_output', x_5by5_flat_plus_cls_output.shape)  # (25+1, b, channel)
        # x_5by5_flat_plus_cls_output_pooling = rearrange(
        #     (x_5by5_flat_plus_cls_output - x_5by5_flat_plus_cls_output[0, :, :]), 'l n c -> n c l ')
        pooling_token_5by5 = reduce(
            x_5by5_flat_plus_cls_output, 'l n c -> n c ',reduction='mean')
        # plt.imshow(x_7by7_flat_plus_cls_output.cpu()[:,0,:])
        # plt.show()
        cls_token_5by5 = x_5by5_flat_plus_cls_output[0, :, :] + self.softmax(cls_token_7by7)
        # plt.plot(cls_token_5by5[0, :].cpu().detach().numpy(), color='orange')
        central_token_5by5 = x_5by5_flat_plus_cls_output[12, :, :] + self.softmax(central_token_7by7)
        # plt.plot(central_token_5by5[0, :].cpu().detach().numpy(), color='red')
        # pooling_token_5by5 = self.adapooling(x_5by5_flat_plus_cls_output_pooling)
        # pooling_token_5by5 = reduce(pooling_token_5by5, 'b c 1 -> b c', reduction='mean')
        pooling_token_5by5 = pooling_token_5by5 + self.softmax(pooling_token_7by7)
        # plt.plot(pooling_token_5by5[0, :].cpu().detach().numpy(), color='blue')
        # plt.show()
        # print('cls:', cls_token_5by5.shape)
        # print('central:', central_token_5by5.shape)
        # print("cls_pooling 2:", self.cos(cls_token_5by5[:,:],pooling_token_5by5[:,:]), "cls_central:",self.cos(cls_token_5by5[:,:],central_token_5by5[:,:]), "pooling_central", self.cos(pooling_token_5by5[:,:],central_token_5by5[:,:]))

        # adaptive pooling
        # cls_token_25_plus_1_output = repeat(cls_token_25_plus_1_output, 'b c -> b () c')
        # central_token = repeat(central_token, 'b c -> b () c')
        # print('cls:', cls_token_25_plus_1_output.shape)
        # print('central:', central_token.shape)
        # adpooling = nn.AdaptiveAvgPool1d(output_size=102).to(
        #     device='cuda')  # output_size根据不同的数据集,需要改动 一半的self.input_channels
        # cls_token_25_plus_1_output = self.adapooling(cls_token_25_plus_1_output)
        # central_token = self.adapooling(central_token)
        # print('cls:', cls_token_25_plus_1_output.shape)
        # print('central:', central_token.shape)
        # cls_token_25_plus_1_output = repeat(cls_token_25_plus_1_output, 'b () c -> b c 5 5')  # 先试试直接变成5by5
        # central_token = repeat(central_token, 'b () c -> b c 5 5')
        # central_token = self.deconv_1to5(central_token)
        # print('cls:', cls_token_25_plus_1_output.shape)
        # print('central:', central_token.shape)
        # 重建成5by5的patch
        x_5by5 = x_5by5_base
        # x_5by5 = torch.cat((cls_token_25_plus_1_output, central_token), dim=1)
        # print('x_5by5', x_5by5.shape)
        #
        '5*5 --conv--> 3*3'

        x_3by3 = self.conv_5to3(x_5by5)
        x_3by3_base = x_3by3
        x_3by3_flat = rearrange(x_3by3, 'b c h w -> b (h w) c')
        # cls token + positional embeding

        b, n, c = x_3by3_flat.shape
        cls_tokens_9_plus_1 = repeat(self.cls_token_9_plus_1, '() n c -> b n c', b=b)
        x_3by3_flat_plus_cls = torch.cat((cls_tokens_9_plus_1, x_3by3_flat), dim=1)
        # print('x_3by3_flat_plus_cls:', x_3by3_flat_plus_cls.shape)

        x_3by3_flat_plus_cls += self.pos_embedding_9_plus_1[:, :(x_3by3_flat_plus_cls.size(1) + 1)]
        # print('x_3by3_flat_plus_cls:', x_3by3_flat_plus_cls.shape)
        # x_3by3_flat_plus_cls = self.dropout(x_3by3_flat)
        # trans setting
        # train trans
        x_3by3_flat_plus_cls = rearrange(x_3by3_flat_plus_cls, 'b n c -> n b c')
        x_3by3_flat_plus_cls_output = self.transformer_encoder_9_plus_1(x_3by3_flat_plus_cls)
        x_3by3_flat_plus_cls_output = self.denselayer(x_3by3_flat_plus_cls_output)
        # print('x_3by3_flat_plus_cls_output', x_3by3_flat_plus_cls_output.shape)  # (9+1, b, channel)
        # x_3by3_flat_plus_cls_output_pooling = reduce((x_3by3_flat_plus_cls_output, 'n b c -> b c'), reduction='mean')
        pooling_token_3by3 = reduce(x_3by3_flat_plus_cls_output, 'n b c -> b c', reduction='mean')
        # plt.imshow(x_7by7_flat_plus_cls_output.cpu()[:,0,:])
        # plt.show()
        # plt.plot(x_3by3_flat_plus_cls_output.permute(2,1,0)[:,0,:].cpu().detach().numpy())
        cls_token_3by3 = x_3by3_flat_plus_cls_output[0, :, :] + self.softmax(cls_token_5by5) + self.softmax(cls_token_7by7)
        # plt.plot(cls_token_3by3[0, :].cpu().detach().numpy(), color='black',linewidth=3)
        central_token_3by3 = x_3by3_flat_plus_cls_output[4, :, :] + self.softmax(central_token_5by5) + self.softmax(central_token_7by7)
        # plt.plot(central_token_3by3[0, :].cpu().detach().numpy(), color='red', linewidth=3)
        # pooling_token_3by3 = self.adapooling(x_3by3_flat_plus_cls_output_pooling)
        # pooling_token_3by3 = reduce(pooling_token_3by3, 'b c 1 -> b c', reduction='mean')
        pooling_token_3by3 = pooling_token_3by3 + self.softmax(pooling_token_7by7) + self.softmax(pooling_token_5by5)
        # plt.plot(pooling_token_3by3[0, :].cpu().detach().numpy(), color='blue',linewidth=3)

        # plt.grid(linewidth=0.5, color='black')
        # plt.title('All tokens', fontdict={'size': 40})
        # plt.xlabel('Spectral size', fontdict={'size': 40}, fontweight='bold')
        # plt.ylabel('Values', fontdict={'size': 40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边

        # plt.show()
        # print("cls_pooling 3:", self.cos(cls_token_3by3[:,:],pooling_token_3by3[:,:]), "cls_central:",self.cos(cls_token_3by3[:,:],central_token_3by3[:,:]), "pooling_central", self.cos(pooling_token_3by3[:,:],central_token_3by3[:,:]))

        # print('cls:', cls_token_9_plus_1_output.shape)
        # print('central:', central_token.shape)
        # adaptive pooling
        # cls_token_9_plus_1_output = repeat(cls_token_9_plus_1_output, 'b c -> b () c')
        # central_token = repeat(central_token, 'b c -> b () c')
        # print('cls:', cls_token_9_plus_1_output.shape)
        # print('central:', central_token.shape)
        # adpooling = nn.AdaptiveAvgPool1d(output_size=102).to(
        #     device='cuda')  # output_size根据不同的数据集,需要改动 一半的self.input_channels
        # cls_token_9_plus_1_output = self.adapooling(cls_token_9_plus_1_output)
        # central_token = self.adapooling(central_token)
        # print('cls:', cls_token_9_plus_1_output.shape)
        # print('central:', central_token.shape)
        # cls_token_9_plus_1_output = repeat(cls_token_9_plus_1_output, 'b () c -> b c 3 3')  # 先试试直接变成5by5
        # central_token = repeat(central_token, 'b () c -> b c 3 3')
        # central_token = self.deconv_1to3(central_token)
        # print('cls:', cls_token_9_plus_1_output.shape)
        # print('central:', central_token_3by3.shape)
        # 重建成3by3的patch
        x_3by3 = x_3by3_base
        # x_3by3 = torch.cat((cls_token_9_plus_1_output, central_token), dim=1)
        # print('x_3by3', x_3by3.shape)

        #
        "3*3 --conv--> 1*1"
        x_1by1 = self.conv_3to1(x_3by3)
        x_1by1_base = self.conv_3to1(x_3by3)
        # print('x_1by1:',x_1by1.shape)
        x_1by1 = rearrange(x_1by1, 'b c () () -> b c')
        # print('x_1by1:', x_1by1.shape)
        central_token_1by1 = x_1by1
        cls_token_1by1 = x_1by1
        pooling_token_1by1 = self.adapooling2d(x_1by1_base)
        pooling_token_1by1 = reduce(pooling_token_1by1, 'b c 1 1 -> b c', reduction='mean')
        #之前是直接加起来，
        central_token_all = central_token_1by1 + self.softmax(central_token_3by3) + self.softmax(central_token_5by5) + self.softmax(central_token_7by7)
        cls_token_all = cls_token_1by1 + self.softmax(cls_token_3by3) + self.softmax(cls_token_5by5) + self.softmax(cls_token_7by7)
        pooling_token_all = pooling_token_1by1 + self.softmax(pooling_token_3by3) + self.softmax(pooling_token_5by5) + self.softmax(pooling_token_7by7)
        # print('central_token_all', central_token_all.shape)
        # plt.plot(cls_token_all[0,:].cpu().detach().numpy(),color = 'orange')
        # plt.plot(central_token_all[0, :].cpu().detach().numpy(), color='red')
        # plt.plot(pooling_token_all[0,:].cpu().detach().numpy(), color='blue')
        # plt.show()
        alpha = self.sigmoid(repeat(self.alpha, '1 1 -> b 1',b=b))
        beta = self.sigmoid(repeat(self.beta, '1 1 -> b 1',b=b))
        gamma = 1 - beta

        # print("cls_pooling 4:", self.cos(cls_token_all[:,:],pooling_token_all[:,:]), "cls_central:",self.cos(cls_token_all[:,:],central_token_all[:,:]), "pooling_central", self.cos(pooling_token_all[:,:],central_token_all[:,:]))
        token_conca_all = torch.cat([alpha * cls_token_all, beta * central_token_all, gamma * pooling_token_all],dim=1)
        token_conca_all = self.bn(token_conca_all)
        token_conca_all = self.dropout(token_conca_all)
        preds = self.fc(token_conca_all)
        # print('alpha:', alpha, 'beta:', beta, 'gamma:', gamma )
        # x_1by1 = self.bn(cls_token_all +central_token_all+pooling_token_all)
        # x_1by1 = self.dropout(x_1by1)
        # preds2 = self.fc(x_1by1)


        "scheme 1"
        # x_strategy_FLC = torch.cat([x1r_output_central,x2r_output_central,x3r_output_central,x4r_output_central,x5r_output_central,x6r_output_central,x7r_output_central,x8r_output_central],dim=0)
        # print('x_strategy_FLC', x_strategy_FLC.shape) # (x, b, c) (8 , batch, 64)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # x_strategy_FLC += self.pos_embedding_conca_scheme1[:, :(x_strategy_FLC.size(1))]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给8个direction
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=32, activation='gelu').to(
        #     device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        #
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds = self.fc_transformer_scheme1(x_strategy_FLC_output)

        "scheme 2"
        # x1_7r = torch.cat([x1r_output_central, x7r_output_central], dim=2 )
        # x2_8r = torch.cat([x2r_output_central, x8r_output_central], dim=2 )
        # x3_5r = torch.cat([x3r_output_central, x5r_output_central], dim=2 )
        # x4_6r = torch.cat([x4r_output_central, x6r_output_central], dim=2 )
        # x_strategy_FLC = torch.cat([x1_7r, x2_8r, x3_5r, x4_6r], dim=0)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # cls_tokens_FLC = repeat(self.cls_token_FLC, '() n c -> b n c', b=b)
        # x_strategy_FLC = torch.cat((cls_tokens_FLC, x_strategy_FLC), dim=1)
        # x_strategy_FLC += self.pos_embedding_conca_scheme2[:, :(x_strategy_FLC.size(1) + 1 )]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给4对directions
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim * 2, nhead=1, dim_feedforward=32,
        #                                                  activation='gelu').to(device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme2(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme2(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds = self.fc_transformer_scheme2(x_strategy_FLC_output)

        return preds