import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.ginconv import GINConvNet
from utils import *
# from un.sub import GraphEnhance
import time
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
from sub import *



# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


# datasets = [['davis','kiba'][int(sys.argv[1])]]
# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# model_st = modeling.__name__


datasets = ['davis']  # ['Kiba']
modeling = GINConvNet
model_st = modeling.__name__

cuda_name = "cuda:0"
# if len(sys.argv)>3:
#     cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

        sub = GraphEnhance(78,128,4,mode='TS',times=2)
        sub.load_state_dict(torch.load('sub_50000.pth'),strict=False)

        # for name,value in sub.named_parameters():
        #     print(name)
        sub.train()

        model = modeling(sub).to(device)
        # model = modeling().to(device)


        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_rm2 = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = '20221228_' + model_st + '_' + dataset +  '.csv'


        with open(result_file_name, 'a') as f:
            f.write('sub_50000_bilstm_1_ngram234_1200_epoch' + '\n')
            f.write('epoch,rmse,mse,pearson,spearman,ci,rm2' + '\n')

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            G,P = predicting(model, device, test_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),get_rm2(G,P)]
            if ret[1]<best_mse:
                torch.save(model.state_dict(), model_file_name)

                with open(result_file_name,'a') as f:
                    f.write(str(epoch+1) + ',' + ','.join(map(str, ret)) + '\n')

                best_epoch = epoch+1
                best_mse = ret[1]
                best_ci = ret[-2]
                best_rm2 = ret[-1]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse,best_ci,best_rm2,model_st,dataset)
            else:
                print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse,best_ci,best_rm2,model_st,dataset)

            epoch_time = time.time() - start_time
            print('epoch_time',epoch_time)

