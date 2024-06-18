import torch.nn as nn
import torch
from torch.nn import init
import numpy as np
from einops import rearrange, repeat, reduce


class ICIP2022(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (
        nn.Linear, nn.GRU, nn.Conv3d, nn.LSTM, nn.Conv2d, nn.TransformerEncoder, nn.TransformerEncoderLayer)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=9, pool='cls', embed_dim=64):
        super(ICIP2022, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.conv1d = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=3, stride=1)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.projection_1d = nn.Linear(90, embed_dim)
        self.projection_3d = nn.Conv3d(in_channels=10, out_channels=16, kernel_size=(1, 2, 2), stride=(1, 1, 1))
        self.lstm_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * (input_channels))
        self.gru_bn_trans = nn.BatchNorm1d(input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_spectralclasstoken = nn.BatchNorm1d(180)
        self.gru_bn_spatialclasstoken = nn.BatchNorm1d(embed_dim)
        self.pos_embedding_spectral = nn.Parameter(torch.randn(1, embed_dim + 1, 180))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, (180) + 1, embed_dim))
        # self.pos_embedding_3d = get_pos_encode2(180, embed_dim, 3, 3, 20)
        # self.pos_embedding_3d = nn.Parameter(torch.randn(1, 12, 272, 3,3))
        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, 180))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.ave1Dpooling_spatial = nn.AdaptiveAvgPool1d(1)
        self.ave1Dpooling_spectral = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(patch_size ** 2 * (input_channels), n_classes)
        self.fc_trans_spatial = nn.Linear(embed_dim, n_classes)
        self.fc_trans_spectral = nn.Linear(180, n_classes)
        self.dense_layer_spectral = nn.Linear(180, 180)
        self.dense_layer_spatial = nn.Linear(embed_dim, embed_dim)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.linearprojection_spatialtrans = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.linearprojection_spectraltrans = nn.Linear(in_features=180, out_features=180)
        self.linearprojection_25_to_81 = nn.Linear(in_features=25, out_features=81)
        self.linearprojection_3d = nn.Linear(in_features=90, out_features=embed_dim)
        self.fc_3 = nn.Linear(64, n_classes)
        self.fc_vit = nn.Linear(input_channels, n_classes)
        self.fc_joint = nn.Linear(n_classes * 2, n_classes)
        self.softmax = nn.Softmax()
        self.pool = pool
        self.layernorm_spe = nn.LayerNorm(180)
        self.layernorm_spa = nn.LayerNorm(embed_dim)
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (c c1) (h h1) (w w1) -> b () (h1 w1 c1)', p1 = 2, p2 = 2))
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

    def partition_3D(self, image, kernel_size, padding, stride):
        """
        Function to partition a 3D image into smaller patches.

        Args:
            image (Tensor): Input image of shape (batch_size, channels, height, width).
            kernel_size (tuple): Size of the patches to create, in the format (height, width, depth).
            padding (tuple): Amount of padding to apply, in the format (height, width, depth).
            stride (tuple): Stride for the patch extraction, in the format (height, width, depth).

        Returns:
            Tensor: Output tensor after partitioning.
        """

        # Check the dimension of the input image
        assert image.dim() == 4, "Input image must be a 4D tensor of shape (batch_size, channels, height, width)."

        # Check the validity of kernel_size, padding, and stride
        # assert all(i <= j for i, j in zip(kernel_size, image.shape[1:])), "Kernel size must be smaller than the image size in all dimensions."

        def partition_along_dimension(tensor, kernel, padding, stride):
            return nn.functional.unfold(tensor, kernel_size=kernel, padding=padding, stride=stride)

        kernel_size_h_w, kernel_size_d = (kernel_size[0], kernel_size[1]), kernel_size[2]
        padding_h_w, padding_d = (padding[0], padding[1]), padding[2]
        stride_h_w, stride_d = (stride[0], stride[1]), stride[2]

        # Partition along height and width
        trans2D = partition_along_dimension(image, kernel_size_h_w, padding_h_w, stride_h_w)
        num_patches = trans2D.shape[-1]
        trans2D = rearrange(trans2D, 'b (c p1 p2) n -> b n (p1 p2) c', p1=kernel_size[0], p2=kernel_size[1])

        # Partition along depth
        trans3D = partition_along_dimension(trans2D, (kernel_size[0] * kernel_size[1], kernel_size_d), (0, padding_d),
                                            stride_d)
        trans3D = rearrange(trans3D, 'b (n2D p p3) n3D -> b (n2D n3D) (p3 p)', p=kernel_size[0] * kernel_size[1],
                            p3=kernel_size[2], n2D=num_patches)
        return trans3D

    def forward(self, x):  # 初始是第1方向
        # print('x.shape', x.shape)

        x = x.squeeze(1)  # B,F,P,P

        # ResNet patch_size = 9 for SA PU
        # x = self.conv2d_1(x)
        # print('1', x.shape)
        # x = self.relu(x)
        # x = self.conv2d_2(x)
        # print('2', x.shape)
        # x_res = self.relu(x)
        # x_res = self.conv2d_3(x_res)
        # print('3', x.shape) #(ptach size = 6)
        # x_res = self.relu(x_res)
        # x_res_res = self.conv2d_4(x_res)
        # x_res_res = self.relu(x_res_res)
        # x = x_res + x_res_res
        # print('4', x.shape) #SA(b,204,6,6)

        # 直接用3DCNN试试
        # x = repeat(x, 'b d h w -> b () d h w')
        # threedcnn_1 = nn.Conv3d(in_channels=1,out_channels=self.embed_dim,kernel_size=3,stride=1).to(device='cuda')
        # y = threedcnn_1(x)
        # print('y',y.shape)#(b, emded__dim, d,4,4)
        # threedcnn_2 = nn.Conv3d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=3, stride=1).to(device='cuda')
        # y = threedcnn_2(y)
        # y = reduce(y,'b c d h w -> b c d 1 1',reduction='mean')
        # y = reduce(y, 'b c d 1 1 -> b c d', reduction='mean')
        # print('y',y.shape)

        # try the 3D overlapped slice
        patch_h = 3
        patch_w = 3
        patch_c = 10
        y = self.partition_3D(x, kernel_size=(patch_h, patch_w, patch_c), padding=(0, 0, 0), stride=(3, 3, 10))
        # print('overlapped3d_x',y.shape)
        y_3d = rearrange(y, 'b num (h w c) -> b c num h w', c=patch_c, h=patch_h, w=patch_w)  # 暂时没用到 (100 10 180 3 3)
        # print('y_3d', y_3d.shape)

        # y = y + self.pos_embedding_3d
        y_1d = self.projection_1d(y)
        # print('y', y.shape)  # (100, 180, 64)
        y_3d = self.projection_3d(y_3d)
        # print('y_3d', y_3d.shape)
        y_3d = y_3d.reshape(y_3d.size(0), y_3d.size(2), -1)
        # y_3d = self.linearprojection_3d(y_3d)
        # print('y_3d', y_3d.shape)

        # pe_h,pe_w,pe_f = self.pos_embedding_3d
        # y_1d = y_1d + pe_h[:,1:181,:].to(device='cuda') + pe_w[:,1:181,:].to(device='cuda') + pe_f[:,1:181,:].to(device='cuda')
        # print('pe_h',pe_h.shape)
        # y = reduce(y, 'b c d h w -> b (c h w) d', reduction='mean')
        # print('overlapped3d_x', y.shape) #SA(b, embed-dim, 3213) 把3213看成步长, embde_dim看成channel所以就是(b,c,x) (100,180,64)
        # try the non-overllaped slice
        # y = rearrange(x, 'b c h w -> b h w c')
        # y = rearrange(y, 'b (h patch_height) (w patch_width) (c patch_channel) -> b (h w c) (patch_height patch_width patch_channel)'
        #               , patch_height=2, patch_width=2, patch_channel=4)
        # y = rearrange(y, 'b num (h1 w1 c1) -> b num h1 w1 c1', h1=2, w1 =2, c1=4) #(b,d,h,w,c) SA:(b,612,3,3,3)
        # print('rearrange', y.shape)
        # y = rearrange(y, 'b d h w c -> b c d h w') #SA(b,3,612,3,3)
        # print('rearrange', y.shape)
        # y = self.projection_3d(y)
        # print('rearrange', y.shape) #(b,c,d h w) SA(b, embed_dim, 204, 1 , 1)
        # y = reduce(y, 'b c d h w -> b (c h w) d', reduction='mean') #SA(b, embed-dim, 204) 把204看成步长, embde_dim看成channel所以就是(b,c,x)
        # print('rearrange', y.shape) #(btach, embed_dim, 204)

        # ResNet patch_size = 5 for IP dataset
        # x_spectraltrans_reconstruct = self.conv2d_5_1(x)
        # x_spectraltrans_reconstruct_1 = x_spectraltrans_reconstruct
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct)
        # x_spectraltrans_reconstruct_2 = self.conv2d_5_2(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.conv2d_5_3(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.relu(x_spectraltrans_reconstruct_2)
        # x = x_spectraltrans_reconstruct_1 + x_spectraltrans_reconstruct_2 + x_spectraltrans_reconstruct_3

        # Transformerencoder 从这里开始
        # 3d spectral trans (SpeViT)
        x_spectraltrans = y_3d.permute(0, 2, 1).contiguous()  # (C,B,X)(100,64,180) 把64当做步长

        # print('x_spectraltrans', x_spectraltrans.shape)

        # spectral Linear Projection
        x_spectraltrans = self.linearprojection_spectraltrans(x_spectraltrans)
        # print('spectral_linearpro1', x_spectraltrans.shape) #SA(100, c64, x180)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C64,B100,X180)

        # spectral cls_token和pos_embedding
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (B100,C64,X180)
        # print('spectral_linearpro2', x_spectraltrans.shape) #SA(100, c64, x180)
        b_spectral, n_spectral, _ = x_spectraltrans.shape
        cls_tokens_spectral = repeat(self.cls_token_spectral,
                                     '() n_spectral d_spectral -> b_spectral n_spectral d_spectral',
                                     b_spectral=b_spectral)
        x_spectraltrans = torch.cat((cls_tokens_spectral, x_spectraltrans), dim=1)
        # print('spectral_linearpro3', x_spectraltrans.shape) #SA(100, c65, x180)
        x_spectraltrans = x_spectraltrans + self.pos_embedding_spectral[:, :(x_spectraltrans.size(1) + 1)]
        # print('spectral_linearpro4', x_spectraltrans.shape) #SA(100, c65, x180)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C65,B100,X180)
        # print('spectral_linearpro5', x_spectraltrans.shape) #SA(c65, b100, x180)

        # 设置spectral transformer参数
        encoder_layer_spectral = nn.TransformerEncoderLayer(d_model=180, nhead=1, dim_feedforward=self.embed_dim,
                                                            activation='gelu').to(device='cuda')
        transformer_encoder_spectral = nn.TransformerEncoder(encoder_layer_spectral, num_layers=1, norm=None).to(
            device='cuda')
        # 最后训练 spectral transformer
        x_spectraltrans_output = transformer_encoder_spectral(x_spectraltrans)
        # print('spectral_linearpro6', x_spectraltrans.shape) #SA(c65, b100, x180) (L,N,C)
        x_spectraltrans_output = self.layernorm_spe(x_spectraltrans_output)
        x_spectraltrans_output = self.dense_layer_spectral(
            x_spectraltrans_output)  # attention之后的全连接层 #SA(c65, b100, x180) (L,N,C)

        x_spectraltrans_output = self.relu(x_spectraltrans_output)  # SA(c65, b100, x180) (L_in,N,C)
        #
        # SpeViT的output layer
        x_spectraltrans_output = rearrange(x_spectraltrans_output,
                                           'length_in batch channel -> batch channel length_in')  # (N,C,L_in)

        x_spectral_classtoken = reduce(x_spectraltrans_output, 'batch channel length_in -> batch channel 1',
                                       reduction='mean')

        # x_spectral_classtoken = self.ave1Dpooling_spectral(x_spectraltrans_output)

        x_spectral_classtoken = self.relu(x_spectral_classtoken)

        # print('spectral_linearpro7', x_spectral_classtoken.shape) #(batch, channel, length)

        x_spectral_classtoken = x_spectral_classtoken.view(x_spectral_classtoken.size(0), -1)
        x_spectral_classtoken = self.gru_bn_spectralclasstoken(x_spectral_classtoken)
        # x_spectral_classtoken = self.prelu(x_spectral_classtoken)
        x_spectral_classtoken = self.dropout(x_spectral_classtoken)
        preds_SpeViT = self.fc_trans_spectral(x_spectral_classtoken)  # SpeViT de output#(SpeViT)

        # ------------------------------------------------------------------分割线----------------------------------------------------------------------------------------------#
        # # 进入3d SpaViT
        x_spatialtrans = y_1d  # (x,b,c) SA(100, 180, 64)

        # spatial Linear Projection
        # x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()
        # print('spatial_linearpro2', x_spatialtrans.shape) #SA(100, 204, embded_dim) (b,x,c)
        x_spatialtrans = self.linearprojection_spatialtrans(x_spatialtrans)  # (100,180,64)
        # print('spatial_linearpro3', x_spatialtrans.shape) #SA(100, 204, 204) (b,x,c)
        # x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()
        x_spatialtrans = x_spatialtrans

        # spatial cls_token和pos_embedding
        # x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (B,X,C)
        b_spatial, n_spatial, _ = x_spatialtrans.shape
        cls_tokens_spatial = repeat(self.cls_token_spatial, '() n_spatial d_spatial -> b_spatial n_spatial d_spatial',
                                    b_spatial=b_spatial)
        x_spatialtrans = torch.cat((cls_tokens_spatial, x_spatialtrans), dim=1)

        # print('spatial_linearpro5', x_spatialtrans.shape)#SA(100, 181,64)

        # x_spatialtrans += pe_h.to(device='cuda')
        # x_spatialtrans += pe_w.to(device='cuda')
        # x_spatialtrans += pe_f.to(device='cuda')

        # plt.subplot(151)
        # plt.imshow(pe_h[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Pos.embedding in height', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)
        #
        # plt.subplot(152)
        # plt.imshow(pe_w[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Pos.embedding in width', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)
        #
        # plt.subplot(153)
        # plt.imshow(pe_f[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Pos.embedding in dimension', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)
        #
        # plt.subplot(154)
        # plt.imshow((pe_f + pe_h + pe_w)[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('sum of pos.embedding', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)

        x_spatialtrans += self.pos_embedding_spatial[:, :(x_spatialtrans.size(1) + 1)]
        # plt.subplot(155)
        # plt.imshow(self.pos_embedding_spatial[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Learned pos.embedding', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)

        # plt.show()
        # print('spatial_linearpro6', x_spatialtrans.shape)#(100,181,64)
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (x181,100,c64)

        # 设置spatial transformer参数
        encoder_layer_spatial = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1,
                                                           dim_feedforward=self.embed_dim,
                                                           activation='gelu').to(device='cuda')
        transformer_encoder_spatial = nn.TransformerEncoder(encoder_layer_spatial, num_layers=1, norm=None).to(
            device='cuda')

        # 最后训练 spatial transformer
        x_spatialtrans_output = transformer_encoder_spatial(x_spatialtrans)  # SA(205,100,204) (x+1, b , c)
        x_spatialtrans_output = self.layernorm_spa(x_spatialtrans_output)

        x_spatialtrans_output = self.dense_layer_spatial(
            x_spatialtrans_output)  # attention之后的全连接层 # SA(205,100,204) (x+1, b , c)

        x_spatialtrans_output = self.relu(x_spatialtrans_output)  # SA(x+1,b,c) (length_in, n, c)

        # 3d SpaViT的output layer
        x_spatialtrans_output = rearrange(x_spatialtrans_output,
                                          'length_in batch channel -> batch channel length_in')  # (N,C,L_in)

        # x_spatial_classtoken = self.ave1Dpooling_spatial(x_spatialtrans_output) #(N, C, 1)
        x_spatial_classtoken = reduce(x_spatialtrans_output, 'batch channel length_in -> batch channel 1',
                                      reduction='mean')

        # print('spatial_linearpro7', x_spatial_classtoken.shape)

        x_spatial_classtoken = self.relu(x_spatial_classtoken)

        # print('spatial_linearpro8', x_spatial_classtoken.shape)

        x_spatial_classtoken = x_spatial_classtoken.view(x_spatial_classtoken.size(0), -1)
        # print('spatial_linearpro9', x_spatial_classtoken.shape)
        x_spatial_classtoken = self.gru_bn_spatialclasstoken(x_spatial_classtoken)
        x_spatial_classtoken = self.dropout(x_spatial_classtoken)
        preds_SpaViT = self.fc_trans_spatial(x_spatial_classtoken)  # SpaViT de output

        return preds_SpaViT
