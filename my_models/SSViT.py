import torch.nn as nn
import torch
from torch.nn import init
import numpy as np

class SSViT(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM,nn.Conv2d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=9, pool = 'cls'):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(SSViT, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, 64, 1, bidirectional=False)
        self.conv1d = nn.Conv1d(in_channels=25,out_channels=25,kernel_size=3,stride=1)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=2,stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.lstm_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.vit = ViT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,channels=input_channels)
        # self.cait = CaiT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,cls_depth=1)
        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_channels,nhead=8)
        self.rnn_2 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        self.gru_3 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=True)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * (input_channels))
        self.gru_bn_trans = nn.BatchNorm1d(input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_spectralclasstoken = nn.BatchNorm1d(input_channels+1)
        # self.layerNorm = nn.LayerNorm((patch_size ** 2) * input_channels)
        self.attention = nn.MultiheadAttention(embed_dim=input_channels,num_heads=1)
        self.attention_2 = nn.MultiheadAttention(embed_dim=64, num_heads=1)
        self.pos_embedding_spectral = nn.Parameter(torch.randn(1, input_channels + 1, 25))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, (25) + 1, input_channels))
        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, 25))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, input_channels))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.ave1Dpooling = nn.AdaptiveAvgPool1d(input_channels)
        self.ave1Dpooling_spectral = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(patch_size**2 * (input_channels), n_classes)
        self.fc_trans_1 = nn.Linear(input_channels, n_classes)
        self.fc_trans_2 = nn.Linear(input_channels+1, n_classes)
        self.dense_layer_spectral = nn.Linear(25, 25)
        self.dense_layer_spatial = nn.Linear(input_channels,input_channels)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.linearprojection_spatialtrans = nn.Linear(in_features=input_channels+1,out_features=input_channels)
        self.linearprojection_spectraltrans = nn.Linear(in_features=25, out_features=25)
        self.linearprojection_25_to_81 = nn.Linear(in_features=25, out_features=81)
        self.fc_3 = nn.Linear(64, n_classes)
        self.fc_vit = nn.Linear(input_channels,n_classes)
        self.fc_joint = nn.Linear(n_classes*2,n_classes)
        self.softmax = nn.Softmax()
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, n_classes)
        )
        self.pool = pool
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


    def forward(self, x): #初始是第1方向
        print('x.shape',x.shape)

        x = x.squeeze(1) #B,F,P,P

        #ResNet patch_size = 9 for SA PU
        x = self.conv2d_1(x)
        print('1',x.shape)
        x = self.prelu(x)
        x = self.conv2d_2(x)
        print('2',x.shape)
        x_res = self.prelu(x)
        x_res = self.conv2d_3(x_res)
        print('3',x.shape)
        x_res = self.prelu(x_res)
        x_res_res = self.conv2d_4(x_res)
        x_res_res = self.prelu(x_res_res)
        x =  x_res +x_res_res
        print('4',x.shape)

        # ResNet patch_size = 5 for IP dataset
        # x_spectraltrans_reconstruct = self.conv2d_5_1(x)
        # x_spectraltrans_reconstruct_1 = x_spectraltrans_reconstruct
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct)
        # x_spectraltrans_reconstruct_2 = self.conv2d_5_2(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.conv2d_5_3(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.relu(x_spectraltrans_reconstruct_2)
        # x = x_spectraltrans_reconstruct_1 + x_spectraltrans_reconstruct_2 + x_spectraltrans_reconstruct_3



        'horizontal'
        x1 = x
        x_horizontal = x1.reshape(x1.shape[0], x1.shape[1], -1)

        x_horizontal = x_horizontal.permute(2, 0, 1)


        '改变scanning的模式'
        x = x_horizontal #(X,B,C) (25,2,204)
        # x_show = x.permute(2,1,0).contiguous()
        # plt.subplot(121)
        # plt.plot(x[:, 0, :].cpu().detach().numpy())
        # plt.plot(x[:, 0, 0].cpu().detach().numpy(),linewidth=5)
        # plt.plot(x[:, 0, 203].cpu().detach().numpy(), linewidth=5)
        # plt.subplot(122)
        # plt.imshow(x[:, 0, :].cpu().detach().numpy())
        print('x.sequence',x.shape) #(seq_len,batch,feature_dimension)
        # plt.show()
        # plt.subplot(211)
        # plt.imshow(x[:, 0, :].cpu().detach().numpy(), cmap='gray')
        # plt.subplot(212)
        # plt.plot(x_show[:, 0, :].cpu().detach().numpy())
        # plt.show()

        #Transformerencoder 从这里开始
        #spectral trans (SpeViT)
        x_spectraltrans = x.permute(2,1,0).contiguous() #(C,B,X)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous() #(B,C,X)
        x_spectraltrans = self.linearprojection_spectraltrans(x_spectraltrans)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()#(C,B,X)

        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (B,C,X)
        b , n , _ = x_spectraltrans.shape
        cls_tokens_spectral =  repeat(self.cls_token_spectral, '() n d -> b n d', b = b)
        x_spectraltrans = torch.cat((cls_tokens_spectral, x_spectraltrans), dim=1)
        x_spectraltrans += self.pos_embedding_spectral[:, :(x_spectraltrans.size(1) + 1)]
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C,B,X)
        print('222',x_spectraltrans.shape)

        encoder_layer_spectral = nn.TransformerEncoderLayer(d_model=25, nhead=1, dim_feedforward=32,activation='gelu').to(device='cuda')
        transformer_encoder_spectral = nn.TransformerEncoder(encoder_layer_spectral, num_layers=1, norm=None).to(device='cuda')

        x_spectraltrans_output = transformer_encoder_spectral(x_spectraltrans) #(C,B,X)(205,b,25)

        x_show_2 = x_spectraltrans_output.permute(2,1,0).contiguous() #(25, b, 205)

        x_pool = reduce(x_show_2, 'x b c -> x b', reduction='mean')
        print('x_pool',x_pool.shape)
        x_cls = x_show_2[:, :, 0]
        print('x_cls',x_cls.shape)
        plt.plot(x_show_2[:,0,:].cpu().detach().numpy())
        plt.plot(x_pool[:,0].cpu().detach().numpy(),color='blue', linewidth=5)
        plt.plot(x_cls[:, 0].cpu().detach().numpy(), color='red', linewidth=5)

        plt.grid(linewidth=0.5, color='black')
        plt.title('All tokens', fontdict={'size': 40})
        plt.xlabel('Spatial size', fontdict={'size': 40}, fontweight='bold')
        plt.ylabel('Values', fontdict={'size': 40})
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)
        bwith = 2  # 边框宽度设置为2
        TK = plt.gca()  # 获取边框
        TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        TK.spines['left'].set_linewidth(bwith)  # 图框左边
        TK.spines['top'].set_linewidth(bwith)  # 图框上边
        TK.spines['right'].set_linewidth(bwith)  # 图框右边

        plt.show()

        x_spectraltrans_output = self.dense_layer_spectral(x_spectraltrans_output) #attention之后的全连接层
        x_spectraltrans_output = self.relu(x_spectraltrans_output)

        #SpeViT的output layer
        x_spectral_classtoken = self.ave1Dpooling_spectral(x_spectraltrans_output)

        print('spectraltoken',x_spectral_classtoken.shape) #(205,b,1)


        # x_spectral_classtoken = x_spectraltrans_output[:,:,0]
        x_spectral_classtoken = x_spectral_classtoken.permute(1, 0, 2).contiguous()
        x_spectral_classtoken = x_spectral_classtoken.view(x_spectral_classtoken.size(0), -1)
        x_spectral_classtoken = self.gru_bn_spectralclasstoken(x_spectral_classtoken)
        # x_spectral_classtoken = self.prelu(x_spectral_classtoken)
        x_spectral_classtoken = self.dropout(x_spectral_classtoken)
        preds_SpeViT = self.fc_trans_2(x_spectral_classtoken) #SpeViT de output
        # x_spectraltrans_output = x_spectraltrans_output.permute()
        # print('x_spectral_out', x_spectraltrans_output.shape) #(200+1,2,25)
        # x_spectraltrans_reconstruct = x_spectraltrans_output.reshape(x_spectraltrans_output.shape[1],x_spectraltrans_output.shape[0],5,5)

        #进入spatial trans (SpaViT)
        x_spatialtrans = x_spectraltrans_output.permute(2,1,0) #(x,b,c)
        print('111',x_spatialtrans.shape) #(25,2,205)
        # x_spatialtrans = self.gelu(x_spatialtrans)

        x_spatialtrans = x_spatialtrans

        # plt.subplot(413)
        # plt.imshow(x_spatialtrans[:, 0, :].cpu().detach().numpy(), cmap='gray')

        #sin and cos positional embedding
        #spatial trans 基于step的position embedding
        # x_pos_spatialtrans = x.permute(1,2,0).contiguous()#B,C,X
        # pos_encoder = PositionalEncoding1D(26).to(device='cuda')
        # x_pos_spatialtrans = pos_encoder(x_pos_spatialtrans)
        # x_pos_spatialtrans = x_pos_spatialtrans.permute(2,0,1).contiguous() #(x,b,c) #(25,100,204)
        # print('x_pos_spatialtrans',x_pos_spatialtrans.shape)

        #spatial trans 基于channel的positional embeddimg
        # x_pos_2_spatialtrans = x.permute(1,0,2).contiguous() #B,X,C
        # pos_encoder_2 = PositionalEncoding1D(204).to(device='cuda')
        # x_pos_2_spatialtrans = pos_encoder_2(x_pos_2_spatialtrans)
        # x_pos_2_spatialtrans = x_pos_2_spatialtrans.permute(1,0,2).contiguous() #(x,b,c) (25,100,204)
        # print('0000', x_pos_2_spatialtrans.shape)

        #spatial Linear Projection
        x_spatialtrans = x_spatialtrans.permute(1,0,2).contiguous()
        print('1111',x_spatialtrans.shape)
        x_spatialtrans = self.linearprojection_spatialtrans(x_spatialtrans)
        x_spatialtrans = x_spatialtrans.permute(1,0,2).contiguous()
        x_spatialtrans = x_spatialtrans  #加不加position encoding?  #(25,100,204)

        #cls_token和pos_embedding
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous() #(B,X,C)
        b, n, _ = x_spatialtrans.shape
        cls_tokens_spatial = repeat(self.cls_token_spatial, '() n d -> b n d', b=b)
        x_spatialtrans = torch.cat((cls_tokens_spatial, x_spatialtrans), dim=1)
        print('xxx',x_spatialtrans.shape)
        x_spatialtrans += self.pos_embedding_spatial[:, :(x_spatialtrans.size(1) + 1)]
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (X,B,C)
        print('xxx',x_spatialtrans.shape)

        # 设置transformer参数
        encoder_layer_spatial = nn.TransformerEncoderLayer(d_model=204, nhead=1, dim_feedforward=32,activation='gelu').to(device='cuda')
        transformer_encoder_spatial = nn.TransformerEncoder(encoder_layer_spatial, num_layers=1, norm=None).to(device='cuda')

        # 最后训练 spatial transformer
        x_spatialtrans_output = transformer_encoder_spatial(x_spatialtrans) # (25,100,204)
        x_spatialtrans_output = self.dense_layer_spatial(x_spatialtrans_output) #attention之后的全连接层
        x_spatialtrans_output = self.relu(x_spatialtrans_output)
        # plt.subplot(414)
        # plt.imshow(x_spatialtrans_output[:, 0, :].cpu().detach().numpy(), cmap='gray')

        #transformer的output进行变形 再卷积 再transformer
        # x_trans_2 = x_trans.reshape(x_trans.shape[1],x_trans.shape[2],5,5)
        # x_trans_2 = self.conv2d_4(x_trans_2)
        # x_trans_2 = self.relu(x_trans_2) #(B,C,H,W)
        # print('x_tran_2',x_trans_2.shape)
        # x_2 = x_trans_2.reshape(-1,x_trans_2.shape[0], x_trans_2.shape[1])
        # print('1',x_2.shape) #S,B,C
        # x_pos_2 = x_2.permute(1, 0, 2).contiguous()
        # pos_encoder_2 = PositionalEncoding1D(512).to(device='cuda')
        # x_pos_2 = pos_encoder_2(x_pos_2)
        # x_pos_2 = x_pos_2.permute(1, 0, 2).contiguous()
        # print('2',x_pos_2.shape)
        # encoder_layer2 = nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=32, activation='relu').to(
        #     device='cuda')
        # transformer_encoder_2 = nn.TransformerEncoder(encoder_layer2, num_layers=2, norm=None).to(device='cuda')
        # x_trans_2 = transformer_encoder_2(x * x_pos)  # (25,100,200)

        x = x_spatialtrans_output #(N,B,C)
        # x_mlp = x.permute(1,0,2).contiguous() #(B,N,C)
        # print('xxx1', x.shape)
        # x_mlp = x_mlp.mean(dim=1) if self.pool == 'mean' else x_mlp[:, 0]
        # print('xxx2', x_mlp.shape)
        # x_mlp = self.to_latent(x_mlp)
        # print('xxx3', x_mlp.shape)

        # plt.subplot(414)
        # plt.imshow(x[:, 0, :].cpu().detach().numpy(), cmap='gray')
        # plt.show()

        #选择中心pixel的值
        x = self.ave1Dpooling(x) #(N,B,C)
        x = self.prelu(x)
        x_center = x[13,:,:] #(B,C)
        x_center = x_center.permute(0, 1).contiguous()
        x = x_center.view(x_center.size(0), -1)
        x = x.view(x.size(0), -1)


        #batchnorm+fc 分类工作
        x = self.gru_bn_trans(x)
        x = self.prelu(x)
        x = self.dropout(x)
        preds_SpaViT = self.fc_trans_1(x)


        preds_joint = torch.cat([preds_SpeViT,preds_SpaViT],dim=1)
        print('preds_joint',preds_joint.shape)
        preds_joint = self.fc_joint(preds_joint)

        # x = self.gru_bn(x)
        # x = self.fc(x)
        # print('preds',x.shape)

        #作图
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
        return  preds_joint + preds_SpeViT + preds_SpaViT