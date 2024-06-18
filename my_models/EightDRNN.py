import torch.nn as nn
import torch
from torch.nn import init
import numpy as np
from einops import rearrange, repeat, reduce

class EightDRNN(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.LSTM, nn.GRU, nn.Parameter)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5, emb_size = 64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(EightDRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        self.gru_3 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=True)
        self.pre_emd = nn.Linear(input_channels, emb_size)
        # self.gru_3_1 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_2 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_3 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_4 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_5 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_6 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_7 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_8 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_1 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_3 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_4 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_5 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_6 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_7 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_8 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_1 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_2 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_3 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_4 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_5 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_6 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_7 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_8 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_1 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_2 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_3 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_4 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_5 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_6 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_7 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_8 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.alpha_1 = nn.Parameter(torch.randn(1, 1))
        self.alpha_2 = nn.Parameter(torch.randn(1, 1))
        self.alpha_3 = nn.Parameter(torch.randn(1, 1))
        self.alpha_4 = nn.Parameter(torch.randn(1, 1))
        self.alpha_5 = nn.Parameter(torch.randn(1, 1))
        self.alpha_6 = nn.Parameter(torch.randn(1, 1))
        self.alpha_7 = nn.Parameter(torch.randn(1, 1))
        self.alpha_8 = nn.Parameter(torch.randn(1, 1))
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * input_channels)
        self.gru_bn_3 = nn.BatchNorm1d(emb_size *(patch_size ** 2))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * input_channels, n_classes)
        self.fc_3 = nn.Linear(emb_size *(patch_size ** 2) , n_classes)
        self.reg = nn.Linear(emb_size *(patch_size ** 2) , input_channels)
        self.softmax = nn.Softmax()
        self.point_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.depth_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, groups=input_channels)
        # self.hyper = HypergraphConv(emb_size, emb_size)
        self.num_hper_e = nn.Parameter(torch.randn(1))
        self.aux_loss_weight = 1

    def forward(self, x): #初始是第1方向
        # x1_1 = self.conv_1x1(x)
        # x3_3 = self.conv_3x3(x)
        # x5_5 = self.conv_5_5(x)
        # x = torch.cat([x3_3, x5_5, x1_1], dim=1)
        x = x.squeeze(1)
        # vis.images(x[1, [12,42,55], :, :],opts={'title':'image'})

        # x = self.relu(self.point_conv_1(x)) + self.relu(self.depth_conv_1(x)) + x

        # vis.images(x[1, [12,42,55], :, :],opts={'title':'image'})
        # print('0', x.shape)
        x1 = x
        x1r = x1.reshape(x1.shape[0], x1.shape[1], -1)


        # x2 = Variable(x1r.cpu())
        # x2 = Variable(x1r).cpu()
        x2 = x1r.cpu()
        x2rn = np.flip(x2.detach().numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()

        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)

        # x4 = Variable(x3r.cpu())
        # x4 = Variable(x3r).cpu()
        x4 = x3r.cpu()
        x4rn = np.flip(x4.detach().numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()

        x5 = torch.rot90(x1, 1, (2, 3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)

        # x6 = Variable(x5r.cpu())
        # x6 = Variable(x5r).cpu()
        x6 = x5r.cpu()
        x6rn = np.flip(x6.detach().numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()

        x7 = torch.transpose(x5, 2, 3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)

        # x8 = Variable(x7r.cpu())
        # x8 = Variable(x7r).cpu()
        x8 = x7r.cpu()
        x8rn = np.flip(x8.detach().numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()

        x8r = x8r.permute(2, 0, 1)
        x7r = x7r.permute(2, 0, 1)
        x6r = x6r.permute(2, 0, 1)
        x5r = x5r.permute(2, 0, 1)
        x4r = x4r.permute(2, 0, 1)
        x3r = x3r.permute(2, 0, 1)
        x2r = x2r.permute(2, 0, 1)
        x1r = x1r.permute(2, 0, 1)

        x1r = self.pre_emd(x1r)
        x2r = self.pre_emd(x2r)
        x3r = self.pre_emd(x3r)
        x4r = self.pre_emd(x4r)
        x5r = self.pre_emd(x5r)
        x6r = self.pre_emd(x6r)
        x7r = self.pre_emd(x7r)
        x8r = self.pre_emd(x8r)

        # print('x1r shape_for mask', x1r.shape)

        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_3 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_4 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_4 = nn.TransformerEncoder(encoder_layer_4, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_5 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_5 = nn.TransformerEncoder(encoder_layer_5, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_6 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_6 = nn.TransformerEncoder(encoder_layer_6, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_7 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_7 = nn.TransformerEncoder(encoder_layer_7, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_8 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_8 = nn.TransformerEncoder(encoder_layer_8, num_layers=1, norm=None).to(device='cuda')

        # 'soft mask with multiscanning'
        # def softweight(x):
        #     x_weight = rearrange(x, 'x b c -> b x c')
        #     x_dist = torch.cdist(x_weight, x_weight, p=2)
        #     mean_x_dist = torch.mean(x_dist)
        #     x_weight_1 = torch.exp(-(x_dist ** 2) / 2 * (mean_x_dist ** 2))
        #     # g = sns.heatmap(x_weight_1[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        #     # g.set_title('')
        #     # plt.show()
        #
        #     mask = np.zeros_like(x_weight_1[1,:,:].cpu().detach().numpy())
        #     mask[np.triu_indices_from(mask)] = True
        #     return x_weight_1
        #
        # mask1 = np.zeros_like(softweight(x1r)[1, :, :].cpu().detach().numpy())
        # mask1[np.triu_indices_from(mask1)] = True
        # plt.subplot(241)
        # g1 = sns.heatmap(softweight(x1r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,mask=mask1)
        # plt.subplot(242)
        # g2 = sns.heatmap(softweight(x2r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(243)
        # g3 = sns.heatmap(softweight(x3r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(244)
        # g4 = sns.heatmap(softweight(x4r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(245)
        # g5 = sns.heatmap(softweight(x5r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(246)
        # g6 = sns.heatmap(softweight(x6r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(247)
        # g7 = sns.heatmap(softweight(x7r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(248)
        # g8 = sns.heatmap(softweight(x8r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.show()


        # '做个spectral soft attention, soft attention mask 速度会变慢'
        # x1r_weight = rearrange(x1r, 'x b c -> b x c')
        # x_dist_1 = torch.cdist(x1r_weight, x1r_weight, p=2)
        # mean_x_dist_1 = torch.mean(x_dist_1)
        # # sns.heatmap(x_dist_1[1, :, :].cpu().detach().numpy(),cmap='Blues',square=True)
        # # # plt.colorbar()
        # # plt.show()
        # x_weight_1 = torch.exp(-(x_dist_1 ** 2) / 2 * (mean_x_dist_1 ** 2))
        # # print('x_weight',x_weight_1.shape)
        # # x_weight = repeat(x_weight,'')
        # # weight = self.sigmoid(weight) * 2
        # g = sns.heatmap(x_weight_1[1, :, :].cpu().detach().numpy(), cmap='Blues',square=True)
        # g.set_title('weight_1')
        # # plt.imshow(weight[1, :, :].cpu().detach().numpy(),cmap='blues')
        # # # plt.colorbar()
        # plt.show()

        # print('x1r',x1r.shape)
        # x = torch.cat([x1r, x1r, x1r, x1r, x1r, x1r, x1r, x1r], dim=2)
        # x1r = self.gru(x1r)[0]

        '---------------hypergraph-----------------'
        # def hyper(x_out):
        #     for b in range(x_out.size(1)):
        #         X = x_out[:,b,:]
        #         hg = dhg.Hypergraph.from_feature_kNN(X, k=5)
        #         X_ = hg.smoothing_with_HGNN(X)
        #         Y_b = hg.v2e(X_,aggr='mean')
        #         X_b = hg.e2v(Y_b,aggr='mean')
        #         if b == 0:
        #             X_new = X_b.unsqueeze(1)
        #         else:
        #             X_new = torch.cat([X_new,X_b.unsqueeze(1)],dim=1)
        #         b = b + 1
        #     print('x_new_2',X_new.shape)
        #     return X_new


        '--------------------------------------------------------------------------------------'
        x1r_out = self.gru_3_1(x1r)
        # print('x1r_output.shape', x1r_out.shape)
        # print('x1r out', x1r_out.shape)  # （25,100,25）
        # print('x1r hidden',x1r_hidden.shape) #（1,100,25）
        # x1r_laststep = x1r_out[-1] #（100,50)
        # print('x1r laststep',x1r_laststep.shape)

        # 'calculate cosine similarity 1'
        # input1 = x1r_out[:,1,:]
        # input_last1 = x1r_laststep[1,:]
        # input_last1 = input_last1.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output1 = pairdistance(input1,input_last1) * (-1)
        # output1 = self.softmax(output1)
        # output1 = output1.unsqueeze(0)
        # # plt.plot(output1[0,:].cpu().detach().numpy(), linewidth =1.5)
        # # plt.show()
        # # sns.heatmap(data=output1.cpu().detach().numpy(), cmap="Blues", linewidths=0.2)
        # # plt.show()
        gamma_1 = self.sigmoid(self.alpha_1)
        delta_1 = 1 - gamma_1
        # x1r_out = transformer_encoder_1(delta_1 * x1r_out + gamma_1 * x1r)
        x1r_out = transformer_encoder_1(x1r_out + x1r) + x1r_out
        # x1r_out = hyper(x1r_out).to(device='cuda')
        # print('x1r_output2.shape', x1r_out.shape) #(step, batch , fea dim)
        # print('gamma_1:', gamma_1, 'delta_1:', delta_1)
        '--------------------------------------------------------------------------------------'
        x2r_out = self.gru_3_2(x2r) #把x1r经过RNN的值，作为x2r的输入
        # x2r_laststep = x2r_out[-1]
        # plt.subplot(2, 4, 2)
        # plt.plot(x2r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 2)
        # plt.plot(x2r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 2'
        # input2 = x2r_out[:, 1, :]
        # input_last2 = x2r_laststep[1, :]
        # input_last2 = input_last2.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output2 = pairdistance(input2, input_last2) * (-1)
        # output2 = self.softmax(output2)
        # output2 = output2.unsqueeze(0)
        gamma_2 = self.sigmoid(self.alpha_2)
        delta_2 = 1 - gamma_2
        # x2r_out = transformer_encoder_2(delta_2 * x2r_out+ gamma_2 * x2r)
        x2r_out = transformer_encoder_2(x2r_out + x2r) +x2r_out
        # x2r_out = hyper(x2r_out).to(device='cuda')
        # print('gamma_2:', gamma_2, 'delta_2:', delta_2)
        '-----------------------------------------------------------------------------------------'
        x3r_out = self.gru_3_3(x3r)
        # x3r_laststep = x3r_out[-1]
        # plt.subplot(2, 4, 3)
        # plt.plot(x3r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 3)
        # plt.plot(x3r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 3'
        # input3 = x3r_out[:, 1, :]
        # input_last3 = x3r_laststep[1, :]
        # input_last3 = input_last3.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output3 = pairdistance(input3, input_last3) * (-1)
        # output3 = self.softmax(output3)
        # output3 = output3.unsqueeze(0)
        gamma_3 = self.sigmoid(self.alpha_3)
        delta_3 = 1 - gamma_3
        # x3r_out = transformer_encoder_3(delta_3 * x3r_out + gamma_3 * x3r)
        x3r_out = transformer_encoder_3(x3r_out + x3r) + x3r_out
        # x3r_out = hyper(x3r_out).to(device='cuda')
        # print('gamma_3:', gamma_3, 'delta_3:', delta_3)
        '----------------------------------------------------------------------------------------'
        x4r_out = self.gru_3_4(x4r)
        # x4r_laststep = x4r_out[-1]
        # plt.subplot(2, 4, 4)
        # plt.plot(x4r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 4)
        # plt.plot(x4r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 4'
        # input4 = x4r_out[:, 1, :]
        # input_last4 = x4r_laststep[1, :]
        # input_last4 = input_last4.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output4 = pairdistance(input4, input_last4) * (-1)
        # output4 = self.softmax(output4)
        # output4 = output4.unsqueeze(0)
        gamma_4 = self.sigmoid(self.alpha_4)
        delta_4 = 1 - gamma_4
        # x4r_out = transformer_encoder_4(delta_4 * x4r_out + gamma_4 * x4r)
        x4r_out = transformer_encoder_4(x4r_out + x4r) +x4r_out
        # x4r_out = hyper(x4r_out).to(device='cuda')
        # print('gamma_4:', gamma_4, 'delta_4:', delta_4)
        '------------------------------------------------------------------------------------------'
        x5r_out = self.gru_3_5(x5r)
        # x5r_laststep = x5r_out[-1]
        # plt.subplot(2, 4, 5)
        # plt.plot(x5r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 5)
        # plt.plot(x5r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 5'
        # input5 = x5r_out[:, 1, :]
        # input_last5 = x5r_laststep[1, :]
        # input_last5 = input_last5.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output5 = pairdistance(input5, input_last5) * (-1)
        # output5 = self.softmax(output5)
        # output5 = output5.unsqueeze(0)
        gamma_5 = self.sigmoid(self.alpha_5)
        delta_5 = 1 - gamma_5
        # x5r_out = transformer_encoder_5(delta_5 * x5r_out + gamma_5 * x5r)
        x5r_out = transformer_encoder_5(x5r_out + x5r) +x5r_out
        # x5r_out = hyper(x5r_out).to(device='cuda')
        # print('gamma_5:', gamma_5, 'delta_5:', delta_5)
        '------------------------------------------------------------------------------------------'
        x6r_out= self.gru_3_6(x6r)
        # x6r_laststep = x6r_out[-1]
        # plt.subplot(2, 4, 6)
        # plt.plot(x6r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 6)
        # plt.plot(x6r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 6'
        # input6 = x6r_out[:, 1, :]
        # input_last6 = x6r_laststep[1, :]
        # input_last6 = input_last6.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output6 = pairdistance(input6, input_last6) * (-1)
        # output6 = self.softmax(output6)
        # output6 = output6.unsqueeze(0)
        gamma_6 = self.sigmoid(self.alpha_6)
        delta_6 = 1 - gamma_6
        # x6r_out = transformer_encoder_6(delta_6 * x6r_out + gamma_6 * x6r)
        x6r_out = transformer_encoder_6(x6r_out + x6r) +x6r_out
        # x6r_out = hyper(x6r_out).to(device='cuda')
        # print('gamma_6:', gamma_6, 'delta_6:', delta_6)
        '---------------------------------------------------------------------------------------------'
        x7r_out= self.gru_3_7(x7r)
        # x7r_laststep = x7r_out[-1]
        # plt.subplot(2, 4, 7)
        # plt.plot(x7r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 7)
        # plt.plot(x7r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 7'
        # input7 = x7r_out[:, 1, :]
        # input_last7 = x7r_laststep[1, :]
        # input_last7 = input_last7.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output7 = pairdistance(input7, input_last7) * (-1)
        # output7 = self.softmax(output7)
        # output7 = output7.unsqueeze(0)
        gamma_7 = self.sigmoid(self.alpha_7)
        delta_7 = 1 - gamma_7
        # x7r_out = transformer_encoder_7(delta_7 * x7r_out + gamma_7 * x7r)
        x7r_out = transformer_encoder_7(x7r_out + x7r) +x7r_out
        # x7r_out = hyper(x7r_out).to(device='cuda')
        # print('gamma_7:', gamma_7, 'delta_7:', delta_7)
        '----------------------------------------------------------------------------------------------'
        x8r_out= self.gru_3_8(x8r)
        # x8r_laststep = x8r_out[-1]
        # ax8 = plt.subplot(2, 4, 8)
        # ax8.set_title('8')
        # plt.plot(x8r_laststep[0,:].cpu().detach().numpy())
        # plt.subplot(1, 8, 8)
        # plt.plot(x8r[:, 0, :].cpu().detach().numpy())
        # # x8r = self.gru(x8r+x7r)[0]
        # print('x8r_out',x8r_out.shape)
        # 'calculate cosine similarity 8'
        # input8 = x8r_out[:, 1, :]
        # input_last8 = x8r_laststep[1, :]
        # input_last8 = input_last8.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output8 = pairdistance(input8, input_last8) * (-1)
        # output8 = self.softmax(output8)
        # output8 = output8.unsqueeze(0)
        gamma_8 = self.sigmoid(self.alpha_8)
        delta_8 = 1 - gamma_8
        # x8r_out = transformer_encoder_8(delta_8 * x8r_out + gamma_8 * x8r)
        x8r_out = transformer_encoder_8(x8r_out + x8r) +x8r_out  # (b n d)
        # x8r_out = hyper(x8r_out).to(device='cuda')
        # print('gamma_8:', gamma_8, 'delta_8:', delta_8)
        step = int(x8r_out.size(0)-1)
        '-------------------------------------------------------------------------------'
        '----show attetntion function------------------------------------------------------'
        # def showattention(inputseq):
        #     allpixel = inputseq[:, 1, :]
        #     linear1 = nn.Linear(allpixel.size(1), allpixel.size(1)).to(device='cuda')
        #     allpixel = linear1(allpixel)
        #
        #     # centralstep = allpixel[12,:]
        #     # laststep = inputseq[int(step/2), 1, :]
        #     # laststep = linear1(laststep)
        #
        #     # centralstep = allpixel[12,:]
        #     centralstep = allpixel[int(step / 2), :]
        #     # centralstep = linear1(centralstep)
        #
        #     pairdis = nn.PairwiseDistance()
        #     cos = nn.CosineSimilarity(dim=-1)
        #
        #     output = torch.matmul(allpixel, centralstep)
        #     # output = pairdis(allpixel, centralstep) * (-1)
        #     # output = cos(allpixel, centralstep) * (-1)
        #
        #     # output = torch.matmul(allpixel, centralstep)
        #     softmax = nn.Softmax()
        #     output = softmax(output)
        #     output = output.unsqueeze(0)
        #     return output
        #
        # '------------------------------------------------------------------------------------'
        # print('x1r_out.shape',x1r_out.shape)
        # output1_1 = showattention(x1r_out)
        # print('......',output1_1.shape)
        # output1_1_image = reduce(output1_1, 'v (h w) -> h w', h=self.patch_size,reduction='mean')
        # sns.heatmap(data=output1_1_image[:,:].cpu().detach().numpy())
        # plt.show()
        # print('......', output1_1.shape)
        # output2_2 = showattention(x2r_out)
        # output3_3 = showattention(x3r_out)
        # output4_4 = showattention(x4r_out)
        # output5_5 = showattention(x5r_out)
        # output6_6 = showattention(x6r_out)
        # output7_7 = showattention(x7r_out)
        # output8_8 = showattention(x8r_out)
        # # '----------------------------------------------------------------------------'
        # outputall = torch.cat([output1_1, output2_2, output3_3, output4_4, output5_5, output6_6, output7_7, output8_8],dim=0)
        # sns.lineplot(data=outputall.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        # all = sns.heatmap(data=outputall.cpu().detach().numpy(), cmap="Blues", linewidths=0.05)
        # all.set_title('all')
        # plt.show()

        '--------------------------------------------------------------------------------------------------'

        # b = x8r_out.shape[1]

        # decoder_layer_1 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_1 = nn.TransformerDecoder(decoder_layer_1, num_layers=1, norm=None).to(device='cuda')
        # memory_1 = x1r_out
        # target_1 = repeat(x1r_out[12,:,:], 'b c -> 1 b c')
        # x1r_decoder_out = transformer_decoder_1(target_1,memory_1)
        # # plt.subplot(121)
        # # plt.imshow(x1r_output[:,1,:].cpu().detach().numpy())
        # # plt.subplot(122)
        # # plt.imshow(x1r_decoder_out[:,1,:].cpu().detach().numpy())
        # # plt.show()
        # decoder_layer_2 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_2 = nn.TransformerDecoder(decoder_layer_2, num_layers=1, norm=None).to(device='cuda')
        # memory_2 = x2r_out
        # target_2 = repeat(x2r_out[12,:,:], 'b c -> 1 b c')
        # x2r_decoder_out = transformer_decoder_2(target_2, memory_2)
        #
        # decoder_layer_3 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_3 = nn.TransformerDecoder(decoder_layer_3, num_layers=1, norm=None).to(device='cuda')
        # memory_3 = x3r_out
        # target_3 = repeat(x3r_out[12,:,:], 'b c -> 1 b c')
        # x3r_decoder_out = transformer_decoder_3(target_3, memory_3)
        #
        # decoder_layer_4 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_4 = nn.TransformerDecoder(decoder_layer_4, num_layers=1, norm=None).to(device='cuda')
        # memory_4 = x4r_out
        # target_4 = repeat(x4r_out[12,:,:],'b c -> 1 b c')
        # x4r_decoder_out = transformer_decoder_4(target_4, memory_4)
        #
        # decoder_layer_5 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_5 = nn.TransformerDecoder(decoder_layer_5, num_layers=1, norm=None).to(device='cuda')
        # memory_5 = x5r_out
        # target_5 = repeat(x5r_out[12,:,:], 'b c -> 1 b c')
        # x5r_decoder_out = transformer_decoder_5(target_5, memory_5)
        #
        # decoder_layer_6 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_6 = nn.TransformerDecoder(decoder_layer_6, num_layers=1, norm=None).to(device='cuda')
        # memory_6 = x6r_out
        # target_6 = repeat(x6r_out[12,:,:], 'b c -> 1 b c')
        # x6r_decoder_out = transformer_decoder_6(target_6, memory_6)
        #
        # decoder_layer_7 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_7 = nn.TransformerDecoder(decoder_layer_7, num_layers=1, norm=None).to(device='cuda')
        # memory_7 = x7r_out
        # target_7 = repeat(x7r_out[12,:,:], 'b c -> 1 b c')
        # x7r_decoder_out = transformer_decoder_7(target_7, memory_7)
        #
        # decoder_layer_8 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_8 = nn.TransformerDecoder(decoder_layer_8, num_layers=1, norm=None).to(device='cuda')
        # memory_8 = x8r_out
        # target_8 = repeat(x8r_out[12,:,:], 'b c -> 1 b c')
        # x8r_decoder_out = transformer_decoder_8(target_8, memory_8)

        b = x1r_out.size(1)
        # print('b',b)
        alpha_1 = (repeat(self.alpha_1, '1 1 -> b 1',b=b))
        alpha_2 = (repeat(self.alpha_2, '1 1 -> b 1',b=b))
        alpha_3 = (repeat(self.alpha_3, '1 1 -> b 1',b=b))
        alpha_4 = (repeat(self.alpha_4, '1 1 -> b 1',b=b))
        alpha_5 = (repeat(self.alpha_5, '1 1 -> b 1',b=b))
        alpha_6 = (repeat(self.alpha_6, '1 1 -> b 1',b=b))
        alpha_7 = (repeat(self.alpha_7, '1 1 -> b 1',b=b))
        alpha_8 = (repeat(self.alpha_8, '1 1 -> b 1',b=b))
        # print('alpha',alpha_1.shape)
        # alpha = alpha_1 + alpha_2 + alpha_3 + alpha_4 + alpha_5 + alpha_6 + alpha_7 + alpha_8
        # alpha_1 = alpha_1 / alpha
        # alpha_2 = alpha_2 / alpha
        # alpha_3 = alpha_3 / alpha
        # alpha_4 = alpha_4 / alpha
        # alpha_5 = alpha_5 / alpha
        # alpha_6 = alpha_6 / alpha
        # alpha_7 = alpha_7 / alpha
        # alpha_8 = alpha_8 / alpha
        attn_alphs = self.softmax(torch.cat([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8], dim=1))
        # print('attn_alpha',attn_alphs.shape)
        # plt.plot(attn_alphs[1,:].cpu().detach().numpy())
        # plt.show()
        attn_alphs = reduce(attn_alphs,'b s -> s', reduction='mean')
        # alpha_1 = rearrange(attn_alphs[:,0],'')

        # if alpha_1 + alpha_2 + alpha_3 + alpha_4 + alpha_5 + alpha_6 + alpha_7 + alpha_8 == 1:
        x = x8r_out + x7r_out + x6r_out + x5r_out + x4r_out+ x3r_out + x2r_out + x1r_out
        # x = x8r_decoder_out + x7r_decoder_out + x6r_decoder_out + x5r_decoder_out + x4r_decoder_out + x3r_decoder_out + x2r_decoder_out + x1r_decoder_out
        # x = x8r_out * attn_alphs[7]  + x7r_out*attn_alphs[6] + x6r_out*attn_alphs[5] + x5r_out*attn_alphs[4] + x4r_out*attn_alphs[3] + x3r_out*attn_alphs[2] + x2r_out * attn_alphs[1] + x1r_out * attn_alphs[0]

        # print('a1:',alpha_1,'a2:',alpha_2,'a3:',alpha_3,'a4:',alpha_4,'a5:',alpha_5,'a6:',alpha_6,'a7:',alpha_7,'a8:',alpha_8)



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
        x = x.view(x.size(0),-1)
        # print('x',x.shape)

        # x = x4 + x3 + x2 + x1
        # # w1 = x1 / x
        # # w2 = x2 / x
        # # w3 = x3 / x
        # # w4 = x4 / x
        # x = 0.35*x1 + 0.35*x2 + 0.15*x3 +0.15*x4
        # # x = w1*x1 + w2*x2 + w3*x3 + w4*x4
        # print('into gru_bn', x.shape)
        x = self.gru_bn_3(x)
        # x = self.gru_bn2(x)
        # x = self.relu(x)
        x = self.tanh(x)
        # x = self.elu(x)
        # x =self.prelu(x)
        # print('into fc',x.shape)
        x = self.dropout(x)
        x_class = self.fc_3(x)
        x_reg = self.reg(x)
        # plt.show()
        # x = self.fc2(x)
        return x_class, x_reg