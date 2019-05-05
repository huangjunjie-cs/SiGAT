#!/usr/bin/env python
# coding=utf-8
__author__ = 'h12345jack'


import sys
import os
import time
import random
import subprocess
from collections import defaultdict
import argparse

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from logistic_function import logistic_embedding4

from common import DATASET_NUM_DIC
from common import my_decorator
#

from fea_extra import FeaExtra

print = my_decorator(print)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', default='bitcoin_alpha', help='Dataset')
parser.add_argument('--dim', type=int, default=20, help='Embedding Dimension')
parser.add_argument('--fea_dim', type=int, default=20, help='Feature Embedding Dimension')
parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
parser.add_argument('--gpu', default='0', help='GPU')
parser.add_argument('--dropout', type=float, default=0.0, help='Folder k')
parser.add_argument('--k', default=1, help='Folder k')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.cuda = args.cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

NUM_NODE = DATASET_NUM_DIC[args.dataset]
WEIGHT_DECAY = args.weight_decay
NODE_FEAT_SIZE = args.fea_dim
EMBEDDING_SIZE1 = args.dim
CUDA = args.cuda
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DROUPOUT = args.dropout
K = args.k

NEG_LOSS_RATIO = 1

'''
'''

class Encoder(nn.Module):
    """
    Encode features to embeddings
    """

    def __init__(self, features_lists, feature_dim, embed_dim, adj_lists, aggs):
        super(Encoder, self).__init__()

        self.features_lists = features_lists
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggs = aggs

        self.embed_dim = embed_dim
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.cuda()

        in_size = self.feat_dim * 3

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.nonlinear_layer = nn.Sequential(
            nn.Linear(self.feat_dim * (len(adj_lists) + 1), self.feat_dim),
            nn.Tanh(),
            nn.Linear(self.feat_dim, self.embed_dim),

        )
        self.nonlinear_layer.apply(init_weights)


    def forward(self, nodes):
        """
        Generates embeddings for nodes.
        """
        neigh_feats = [agg.forward(nodes, adj) for adj, agg in zip(self.adj_lists, self.aggs)]
        if CUDA:
            self_feats = self.features_lists[0](torch.LongTensor(nodes).cuda())
        else:
            self_feats = self.features_lists[0](torch.LongTensor(nodes))
        combined = torch.cat([self_feats] + neigh_feats, 1)
        combined = self.nonlinear_layer(combined)
        return combined


class AttentionAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, node_num,  dropout_rate=DROUPOUT, slope_ratio=0.1):
        super(AttentionAggregator, self).__init__()

        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))
        nn.init.kaiming_normal_(self.a.data)
        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)


    def forward(self, nodes, adj):
        """
        nodes --- list of nodes in a batch
        adj --- sp.csr_matrix
        """
        node_pku = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_pku[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))

        batch_node_num = len(unique_nodes_list)
        # this dict can map new i to originial node id
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)


        # edges = np.copy(edges1)
        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).cuda()

        new_embeddings = self.out_linear_layer(self.features(n2))

        original_node_edge = np.array([self.unique_nodes_dict[nodes], self.unique_nodes_dict[nodes]]).T
        edges = np.vstack((edges, original_node_edge))

        edges = torch.LongTensor(edges).cuda()

        edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], new_embeddings[edges[:, 1], :]), dim=1)

        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))
        indices = edges

        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
                                         torch.Size([batch_node_num, batch_node_num]))

        row_sum = torch.matmul(matrix, torch.ones(size=(batch_node_num, 1)).cuda())
        edges_h = self.dropout(edges_h)

        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
                                         torch.Size([batch_node_num, batch_node_num]))

        results = torch.matmul(matrix, new_embeddings)


        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).cuda(), row_sum)

        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]


class SiGAT(nn.Module):

    def __init__(self, enc):
        super(SiGAT, self).__init__()
        self.enc = enc
        # self.status_score = nn.Linear(
        #     EMBEDDING_SIZE1 * 2, 1,
        #     nn.Sigmoid()
        # )

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds


    def criterion2(self, nodes, pos_neighbors, neg_neighbors):
        pos_neighbors_list = [set.union(pos_neighbors[i]) for i in nodes]
        neg_neighbors_list = [set.union(neg_neighbors[i]) for i in nodes]
        unique_nodes_list = list(set.union(*pos_neighbors_list).union(*neg_neighbors_list).union(nodes))
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}
        nodes_embs = self.enc(unique_nodes_list)

        loss_total = 0
        for index, node in enumerate(nodes):
            z1 = nodes_embs[unique_nodes_dict[node], :]
            pos_neigs = list([unique_nodes_dict[i] for i in pos_neighbors[node]])
            neg_neigs = list([unique_nodes_dict[i] for i in neg_neighbors[node]])
            pos_num = len(pos_neigs)
            neg_num = len(neg_neigs)

            if pos_num > 0:
                pos_neig_embs = nodes_embs[pos_neigs, :]
                loss_pku = -1 * torch.sum(F.logsigmoid(torch.einsum("nj,j->n", [pos_neig_embs, z1])))
                loss_total += loss_pku
            tmp_pku = 1 if neg_num == 0 else neg_num
            C = pos_num // tmp_pku
            if C == 0:
                C = 1
            if neg_num > 0:
                neg_neig_embs = nodes_embs[neg_neigs, :]
                loss_pku = -1 * torch.sum(F.logsigmoid(-1 * torch.einsum("nj,j->n",[neg_neig_embs, z1])))
                loss_total += C * NEG_LOSS_RATIO  * loss_pku

        return loss_total


def load_data2(filename='', add_public_foe=True):

    adj_lists1 = defaultdict(set)
    adj_lists1_1 = defaultdict(set)
    adj_lists1_2 = defaultdict(set)
    adj_lists2 = defaultdict(set)
    adj_lists2_1 = defaultdict(set)
    adj_lists2_2 = defaultdict(set)
    adj_lists3 = defaultdict(set)


    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            person1 = int(info[0])
            person2 = int(info[1])
            v = int(info[2])
            adj_lists3[person2].add(person1)
            adj_lists3[person1].add(person2)

            if v == 1:
                adj_lists1[person1].add(person2)
                adj_lists1[person2].add(person1)

                adj_lists1_1[person1].add(person2)
                adj_lists1_2[person2].add(person1)
            else:
                adj_lists2[person1].add(person2)
                adj_lists2[person2].add(person1)

                adj_lists2_1[person1].add(person2)
                adj_lists2_2[person2].add(person1)


    return adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3


def read_emb(num_nodes, fpath):
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                dim = int(line.split()[1])
                embeddings = np.random.rand(num_nodes, dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                assert len(emb) == dim
                embeddings[int(node)] = np.array(emb)
    return embeddings

def run_emb(dataset='bitcoin_alpha', k=2):
    num_nodes = DATASET_NUM_DIC[dataset] + 3

    # adj_lists1, adj_lists2, adj_lists3 = load_data(k, dataset)
    filename = './experiment-data/{}-signed-{}.edgelist'.format(dataset, k)
    adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3 = load_data2(filename, add_public_foe=False)
    print(k, dataset, 'data load!')
    features = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    features.weight.requires_grad = True

    if CUDA:
        features.cuda()

    adj_lists = [adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2]


    #######
    fea_model = FeaExtra(dataset=dataset, k=k)
    adj_additions1 = [defaultdict(set) for _ in range(16)]
    adj_additions2 = [defaultdict(set) for _ in range(16)]
    adj_additions0 = [defaultdict(set) for _ in range(16)]
    a, b = 0, 0

    for i in adj_lists3:
        for j in adj_lists3[i]:
            v_list = fea_model.feature_part2(i, j)
            for index, v in enumerate(v_list):
                if v > 0:
                    adj_additions0[index][i].add(j)


    for i in adj_lists1_1:
        for j in adj_lists1_1[i]:
            v_list = fea_model.feature_part2(i, j)
            for index, v in enumerate(v_list):
                if v > 0:
                    adj_additions1[index][i].add(j)
                    a += 1

    for i in adj_lists2_1:
        for j in adj_lists2_1[i]:
            v_list = fea_model.feature_part2(i, j)
            for index, v in enumerate(v_list):
                if v > 0:
                    adj_additions2[index][i].add(j)
                    b += 1
    assert a > 0, 'positive something wrong'
    assert b > 0, 'negative something wrong'

    # 38
    adj_lists = adj_lists + adj_additions1 + adj_additions2
    #adj_lists = adj_lists + adj_additions1 + adj_additions2 + [adj_lists3]
    ########################


    def func(adj_list):
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        edges = np.array(edges)
        adj = sp.csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(num_nodes, num_nodes))
        return adj


    adj_lists = list(map(func, adj_lists))
    features_lists = [features for _ in range(len(adj_lists))]
    print(len(adj_lists), 'attention')
    aggs = [AttentionAggregator(features, NODE_FEAT_SIZE, NODE_FEAT_SIZE, num_nodes) for features, adj in
            zip(features_lists, adj_lists)]

    enc1 = Encoder(features_lists, NODE_FEAT_SIZE, EMBEDDING_SIZE1, adj_lists, aggs)

    model = SiGAT(enc1)
    if CUDA:
        model.cuda()
    print(model.train())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters()) + list(enc1.parameters()) \
                                        + list(features.parameters())),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY
                                 )


    for epoch in range(EPOCHS + 2):
        total_loss = []
        if True:
            model.eval()
            all_embedding = np.zeros((NUM_NODE, EMBEDDING_SIZE1))
            for i in range(0, NUM_NODE, BATCH_SIZE):
                begin_index = i
                end_index = i + BATCH_SIZE if i + BATCH_SIZE < NUM_NODE else NUM_NODE
                values = np.arange(begin_index, end_index)
                embed = model.forward(values.tolist())
                embed = embed.data.cpu().numpy()
                all_embedding[begin_index: end_index] = embed

            np.save('./sigat-results/embedding-{}-{}-{}.npy'.format(dataset, k, str(epoch)), all_embedding)
            model = model.train()

        time1 = time.time()
        nodes_pku = np.random.permutation(NUM_NODE).tolist()
        for batch in range(NUM_NODE // BATCH_SIZE):
            optimizer.zero_grad()
            b_index = batch * BATCH_SIZE
            e_index = (batch + 1) * BATCH_SIZE
            nodes = nodes_pku[b_index:e_index]

            loss = model.criterion2(
                nodes, adj_lists1, adj_lists2
            )
            total_loss.append(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()
        print(epoch, np.sum(total_loss), time.time() - time1)


def main():
    run_emb(dataset=args.dataset, k=K)


# hello


if __name__ == "__main__":
    main()

