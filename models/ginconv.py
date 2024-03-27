from models.transfor import make_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch.nn import LSTM



class Conv1dReLU(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            # nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            #           padding=padding),
            nn.LSTM(in_channels,out_channels,1,bidirectional=True,batch_first=True),
            nn.ReLU()
        )

    def forward(self, x):

        LSTM,_= self.inc[0](x)
        # return self.inc(x)
        return self.inc[1](LSTM)


class StackLayer(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))

        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels*2, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))  # 自适应池化 输出维度为1

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackLayer(block_idx + 1, 128, 64, 3)
                # nn.LSTM(128, 64, 1, batch_first=True, bidirectional=True)
                # 改成bilstm
            )
        # self.linear = nn.Linear(block_num * 96, 96)
        self.linear = nn.Linear(block_num * 1200, 128)

    def forward(self, x):
        # x = self.embed(x).permute(0, 2, 1)  # 512 128 1000
        x = self.embed(x)
        feats = [block(x) for block in self.block_list]   # 512 96   *3
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x


# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, sub,n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)


        self.fc_xt_2 = nn.Linear(125*16,16)
        # self.LSTM_xt_2 = nn.LSTM(embed_dim, 8, 1, batch_first=True, bidirectional=True)
        self.embedding_xt_1 = nn.Embedding(434, embed_dim)
        self.embedding_xt_2 = nn.Embedding(8188, embed_dim)
        self.embedding_xt_3 = nn.Embedding(94816, embed_dim)


        # self.LSTM_xt = nn.LSTM(embed_dim, 64, 1, batch_first=True, bidirectional=True)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

        self.sub = sub.encoder
        self.fc_x = nn.Linear(128,128)
        self.fc_xt = nn.Linear(1000*128, 128)

        # 4 94816
        # self.trans = make_model(8188, N=1, d_model=128, d_ff=512, h=8, dropout=0.1, MAX_LEN=1200)
        # self.trans_out = nn.Linear(128*1000, 128)

        self.protein_encoder_1 = TargetRepresentation(1, 434, 128)  # 434 421
        self.protein_encoder_2 = TargetRepresentation(1, 8188, 128)  # 8188 8044
        self.protein_encoder_3 = TargetRepresentation(1, 94816, 128)  # 94816 74353


        self.linear = nn.Linear(128*3,128)
        # self.pool = nn.AdaptiveMaxPool1d(1)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch


        # target, mask = data.target, data.target_mask
        # n_word = data.n_word[0]

        x = self.sub(x,edge_index,batch,percent=0)
        x = self.fc_x(x)


        target_2,target_3,target_4 = data.target_2,data.target_3,data.target_4

        # embedded_xt_1 = self.embedding_xt_1(target_2)  # 512 1200 128
        xt_1 = self.protein_encoder_1(target_2)

        # embedded_xt_2 = self.embedding_xt_2(target_3)
        xt_2 = self.protein_encoder_2(target_3)

        # embedded_xt_3 = self.embedding_xt_3(target_4)
        xt_3 = self.protein_encoder_3(target_4)

        xt = torch.cat((xt_1, xt_2, xt_3), dim=-1)  # 512 128*3
        xt = self.linear(xt)


        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


