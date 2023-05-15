from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json, os
import numpy as np
import os.path as osp
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from arguments import arg_parse
from model import GraphEnhance
from utils import get_embeddings, move_to
from evaluate_embedding import evaluate_embedding
from torch.utils.tensorboard import SummaryWriter
import pprint as pp
import datetime
# from tool import pd_toExcel
def warn(*args, **kwargs):
    pass


import warnings
from utils import *
warnings.warn = warn



if __name__ == "__main__":
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #     for args.proloss in [True, False]:
    for args.num_gc_layers in [4]:
        pp.pprint(vars(args))
        epochs = 100
        log_interval = 5
        batch_size = 128

        lr = args.lr
        DS = args.DS
        mode = args.mode
        percent = args.percent
        times = args.times

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.proloss:
            pro_loss = 'withProLoss'
        else:
            pro_loss = 'NoProLoss'

        best_list = []
        for i in range(1):
            accuracies = {'logreg': [], 'svc': [], 'linearsvc': [], 'randomforest': []}



            # dataset = TUDataset('../data', name=DS).shuffle()
            # num_features = max(dataset.num_features, 1)
            # dataloader = DataLoader(dataset, batch_size=batch_size)

            dataset = TestbedDataset(root='../data', dataset='zinc_50000').shuffle()
            dataloader = DataLoader(dataset, batch_size=batch_size)  # 在这里按batch将小图封装成大图
            num_features = dataset[0].x.size(-1)

            log_dir = os.path.join(f'log{args.hidden_dim}', DS, '{}'.format(mode),
                                   pro_loss,
                                   'times-{}'.format(times),
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            writer = SummaryWriter(log_dir)

            # print('================')
            # print('lr: {}'.format(lr))
            # print('num_features: {}'.format(num_features))
            # print('hidden_dim: {}'.format(args.hidden_dim))
            # print('num_gc_layers: {}'.format(args.num_gc_layers))
            # print('================')

            device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
            model = GraphEnhance(num_features, args.hidden_dim, args.num_gc_layers, mode, times=times).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # print('\nTraining in {}-th loop'.format(i + 1))
            # try:
            #     with tqdm(range(1, epochs + 1), desc='{}-th loop'.format(i + 1)) as tadm_range:
            #
            nce_data = {'epoch': [], 'nce': []}
            best  = 0
            loss_max = sys.maxsize
            for epoch in tqdm(range(1, epochs + 1), desc='{}-th loop'.format(i + 1), ncols=80):
                loss_all = 0
                E_pos_all = 0
                E_neg1_all = 0
                E_neg2_all = 0
                all_nce = 0
                model.train()
                for data in dataloader:

                    # node_attr, edge_attr, edge_idx, batch, y = move_to(data, device)

                    node_attr, edge_idx, batch, y = move_to(data, device)

                    optimizer.zero_grad()
                    # loss, ProLoss, E_neg1, E_neg2, E_pos, nce = model(node_attr, edge_idx, batch, percent=percent)
                    loss = model(node_attr, edge_idx, batch, percent=percent)
                    # all_nce += nce
                    # if args.proloss:
                    #     loss = loss + ProLoss
                    loss_all += loss.item() * data.num_graphs
                    # E_pos_all += E_pos
                    # E_neg1_all += E_neg1
                    # E_neg2_all += E_neg2
                    loss.backward()
                    optimizer.step()
                print()
                print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))
                s_loss = loss_all / len(dataloader)

                # 保存模型
                if abs(s_loss) < abs(loss_max):

                    torch.save(model.state_dict(), 'sub_50000_gat.pth')
                    loss_max = s_loss

