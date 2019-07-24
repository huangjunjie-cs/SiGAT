#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: h12345jack
@file: logistic_function.py
@time: 2018/12/16
"""

import os
import sys
import re
import time
import json
import pickle
import logging
import math
import random
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import defaultdict

import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn import linear_model
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import Normalizer

from common import DATASET_NUM_DIC
from fea_extra import FeaExtra

EMBEDDING_SIZE = 20

SINE_MODEL_PATH_DIC = {
    'epinions': './embeddings/sine_epinions_models',
    'slashdot': './embeddings/sine_slashdot_models',
    'bitcoin_alpha': './embeddings/sine_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/sine_bitcoin_otc_models'
}

SIDE_MODEL_PATH_DIC = {
    'epinions': './embeddings/side_epinions_models',
    'slashdot': './embeddings/side_slashdot_models',
    'bitcoin_alpha': './embeddings/side_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/side_bitcoin_otc_models'
}


def read_train_test_data(dataset, k):
    train_X = []
    train_y = []
    with open('./experiment-data/{}-train-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            train_X.append((i, j))
            train_y.append(flag)
    test_X = []
    test_y = []
    with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            test_X.append((i, j))
            test_y.append(flag)
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)



def common_logistic(dataset, k, embeddings, model):
    train_X, train_y, test_X, test_y  = read_train_test_data(dataset, k)

    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))


    logistic_function = linear_model.LogisticRegression()
    logistic_function.fit(train_X1, train_y)
    pred = logistic_function.predict(test_X1)
    pred_p = logistic_function.predict_proba(test_X1)


    pos_ratio =  np.sum(test_y) / test_y.shape[0]
    accuracy =  metrics.accuracy_score(test_y, pred)
    f1_score0 =  metrics.f1_score(test_y, pred)
    f1_score1 =  metrics.f1_score(test_y, pred, average='macro')
    f1_score2 =  metrics.f1_score(test_y, pred, average='micro')

    auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    print("pos_ratio:", pos_ratio)
    print('accuracy:', accuracy)
    print("f1_score:", f1_score0)
    print("macro f1_score:", f1_score1)
    print("micro f1_score:", f1_score2)
    print("auc score:", auc_score)

    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2,  auc_score


def logistic_embedding0(k=1, dataset='epinions'):
    """using random embedding to train logistic

    Keyword Arguments:
        k {int} -- [folder] (default: {1})
        dataset {str} -- [dataset] (default: {'epinions'})

    Returns:
        [type] -- [pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score]
    """
    print('random embeddings')
    embeddings = np.random.rand(DATASET_NUM_DIC[dataset], EMBEDDING_SIZE)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'random')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score

def read_emb(fpath, dataset):
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                ll = line.split()
                assert len(ll) == 2, 'First line must be 2 numbers'
                dim = int(ll[1])
                embeddings = np.random.rand(DATASET_NUM_DIC[dataset], dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                embeddings[int(node)] = np.array(emb)
    return embeddings

def logistic_embedding1(k=1, dataset='epinions'):
    """use deepwalk embeddings to train logistic function
    
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """

    fpath = os.path.join('embeddings/deepwalk_emb', '{}-{}.emb'.format(dataset, k))
    embeddings = read_emb(fpath, dataset)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'deepwalk')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score

def logistic_embedding2(k=1, dataset='epinions'):
    """use node2vec embeddings to train logistic function
    
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """

    fpath = os.path.join('embeddings/node2vec_emb', '{}-{}.emb'.format(dataset, k))
    embeddings = read_emb(fpath, dataset)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'node2vec')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score

def logistic_embedding3(k=1, dataset='epinions'):
    """use line embeddings to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """
    fpath = os.path.join('embeddings/line_emb', '{}-{}.emb'.format(dataset, k))
    embeddings = read_emb(fpath, dataset)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'line')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score


def logistic_embedding4(k=1, dataset='epinions', epoch=6, dirname='graphssa-results'):
    """use graphssa to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """
    print('item: graphssa with feo', k, epoch)

    filename = os.path.join(dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
    embeddings = np.load(filename)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'graphssa')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score

def logistic_embedding5(k=1, dataset='epinions', epoch=50, v0=True):
    """use sine embeddings to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """

    print('sine', k, 'v0', v0)
    embeddings = []
    if v0:
        filename = os.path.join(SINE_MODEL_PATH_DIC[dataset], str(k) + 'a', str(epoch) + '.p')
    else:
        filename = os.path.join(SINE_MODEL_PATH_DIC[dataset], str(k) + 'b', str(epoch) + '.p')

    # filename = os.path.join('./models/', str(epoch) + '.p')
    print(filename)
    params = ""
    with open(filename, 'rb') as fp:
        params = pickle.load(fp)
        embeddings = params[0].get_value()
    embeddings = embeddings[1:,]
    print(embeddings.shape)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'sine')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score

def logistic_embedding6(k=1, dataset='epinions', epoch=1):
    """use side embeddings to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """
    def read_side_emb():
        voc_path = os.path.join('embeddings/side', '{}{}.vocab'.format(dataset, k))
        order_dict = defaultdict(int)
        with open(voc_path) as f:
            for index, line in enumerate(f.readlines()):
                num = re.findall(r'b\'(\d+)\'', line)
                order_dict[index] = int("".join(num))
        embeddings = np.zeros((DATASET_NUM_DIC[dataset], 50))
        embed_path = os.path.join('embeddings/side', '{}{}{}.emb'.format(dataset, k, epoch))
        with open(embed_path) as f:
            for i, line in enumerate(f.readlines()):
                line_l = line.split()
                emb = [np.float(j) for j in line_l]
                embeddings[order_dict[i]] = np.array(emb)
        return embeddings

    embeddings = read_side_emb()
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'side')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score

def logistic_embedding7(k=1, dataset='epinions', dirname="sign2vec"):
    """use signet embeddings to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """
    print('signet', k, dataset)
    filename = os.path.join('embeddings', dirname, 'embeddings-{}-{}.npy'.format(dataset, k))
    embeddings = np.load(filename)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'signet')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score

def logistic_embedding8(k=1, dataset='epinions'):
    """use feature to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """
    print(dataset, k, 'fea')
    train_X, train_y, test_X, test_y  = read_train_test_data(dataset, k)
    fea = FeaExtra(k=k, dataset=dataset)
    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(fea.get_features(i, j))

    for i, j in test_X:
        test_X1.append(fea.get_features(i, j))

    logistic = linear_model.LogisticRegression()
    logistic.fit(train_X1, train_y)

    pred = logistic.predict(test_X1)
    pred_p = logistic.predict_proba(test_X1)
    pos_ratio =  np.sum(test_y) / test_y.shape[0]
    accuracy =  metrics.accuracy_score(test_y, pred)
    f1_score0 =  metrics.f1_score(test_y, pred)
    f1_score1 =  metrics.f1_score(test_y, pred, average='macro')
    f1_score2 =  metrics.f1_score(test_y, pred, average='micro')

    auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    print("pos_ratio:", pos_ratio)
    print('accuracy:', accuracy)
    print("f1_score:", f1_score0)
    print("macro f1_score:", f1_score1)
    print("micro f1_score:", f1_score2)
    print("auc score:",auc_score)

    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score


def logistic_embedding9(k=1, dataset='epinions', epoch=10, dirname='sigat'):
    """use sigat embedding to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """

    filename = os.path.join('embeddings', dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
    embeddings = np.load(filename)
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, 'sigat')
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score



def main():
    dataset = 'bitcoin_alpha'
    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = logistic_embedding9(k=2, dataset=dataset, epoch=100, dirname='sigat')
    
    # print("pos_ratio:", pos_ratio)
    # print('accuracy:', accuracy)
    # print("f1_score:", f1_score0)
    # print("macro f1_score:", f1_score1)
    # print("micro f1_score:", f1_score2)
    # print("auc score:",auc_score)
        


if __name__ == "__main__":
    main()
