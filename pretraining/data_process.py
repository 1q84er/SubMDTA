import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)  # mol格式
    
    c_size = mol.GetNumAtoms()  # 原子个数
    
    features = []  # 原子个数 features：78
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  # 按照元素有无,度???进行编码
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()  # 有向图
    edge_index = []  #
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index  # 分子大小，原子编码，边关系（有序



if __name__ == '__main__':

    zinc_smiles = []
    with open('zinc.txt') as f:
        zinc_data = f.readlines()
        for item in zinc_data:
            zinc_smiles.append(item[:-1])
    import random

    random.seed(1234)
    num = random.sample(range(0, len(zinc_smiles)), 50000)
    smiles = [zinc_smiles[i] for i in num]

    print('drug number',len(set(smiles)))  # 数据集总共的smiles
    smile_graph = {}  # 将smiles转成分子图
    for smile in smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    print('smile to graph done')
    # convert to PyTorch data format

    processed_data_file = '../data/processed/' + 'zinc_50000.pt'
    if not os.path.isfile(processed_data_file):
        # df = pd.read_csv('data/' + dataset + '_train.csv') train_drugs, train_prots,  train_Y = list(df[
        # 'compound_iso_smiles']),list(df['target_sequence']),list(df['affinity']) XT = [seq_cat(t) for t in train_prots]
        # train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)  # smiles,
        # 蛋白质编码，预测值
        drugs = np.asarray(smiles)
        # make data PyTorch Geometric ready
        print('preparing zinc.pt in pytorch format!')
        train_data = TestbedDataset(root='../data', dataset='zinc_50000', xd=drugs, smile_graph=smile_graph)
    else:
        print(processed_data_file, ' are already created')
