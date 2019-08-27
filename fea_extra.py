#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: huangjunjie
@file: fea_extra.py
@time: 2018/12/10
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

from collections import defaultdict

import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn import linear_model
from sklearn import metrics

from common import DATASET_NUM_DIC


class FeaExtra(object):

    def __init__(self, dataset='epinions', k=1, debug=False):
        filename = './experiment-data/{}-train-{}.edgelist'.format(dataset, k)
        if debug:
            filename = './test.edgelists'
        res = self.init_edgelists(filename=filename)
        self.pos_in_edgelists, self.pos_out_edgelists, self.neg_in_edgelists, self.neg_out_edgelists = res

    def init_edgelists(self, filename='./experiment-data/epinions-train-1.edgelist'):
        
        pos_out_edgelists = defaultdict(list)
        neg_out_edgelists = defaultdict(list)
        pos_in_edgelists = defaultdict(list)
        neg_in_edgelists = defaultdict(list)
        with open(filename) as f:
            for line in f.readlines():
                x, y, z = line.split()
                x = int(x)
                y = int(y)
                z = int(z)
                if z == 1:
                    pos_out_edgelists[x].append(y)
                    pos_in_edgelists[y].append(x)
                else:
                    neg_out_edgelists[x].append(y)
                    neg_in_edgelists[y].append(x)
        return pos_in_edgelists, pos_out_edgelists, neg_in_edgelists, neg_out_edgelists

    def get_pos_indegree(self, v):
        return len(self.pos_in_edgelists[v])

    def get_pos_outdegree(self, v):
        return len(self.pos_out_edgelists[v])

    def get_neg_indegree(self, v):
        return len(self.neg_in_edgelists[v])

    def get_neg_outdegree(self, v):
        return len(self.neg_out_edgelists[v])

    def common_neighbors(self, u, v):
        u_neighbors = self.pos_in_edgelists[u] + self.neg_in_edgelists[u] + \
                      self.pos_out_edgelists[u] + self.neg_out_edgelists[u]
        v_neighbors = self.pos_in_edgelists[v] + self.neg_in_edgelists[v] + \
                      self.pos_out_edgelists[v] + self.neg_out_edgelists[v]
        return len(set(u_neighbors).intersection(set(v_neighbors)))

    def feature_part1(self, u, v):
        d_pos_in_u = self.get_pos_indegree(u)
        d_neg_in_v = self.get_neg_indegree(v)
        d_pos_out_u = self.get_pos_outdegree(u)
        d_neg_out_v = self.get_neg_outdegree(v)

        # d_pos_in_v = self.get_pos_indegree(v)
        # d_neg_in_u = self.get_neg_indegree(u)
        # d_pos_out_v = self.get_pos_outdegree(v)
        # d_neg_out_u = self.get_neg_outdegree(u)

        c_u_v = self.common_neighbors(u, v)
        d_out_u = self.get_neg_outdegree(u) + self.get_pos_outdegree(u)
        d_in_v = self.get_neg_indegree(v) + self.get_pos_indegree(v)
        return d_pos_in_u, d_neg_in_v, d_pos_out_u, d_neg_out_v, c_u_v, d_out_u, d_in_v

    def feature_part2(self, u, v):
        """
        /^ \v /^ \^ /v \v /v ^\
        ++
        /^ \v /^ \^ /v \v /v ^\
        +-
        /^ \v /^ \^ /v \v /v ^\
        -+
        /^ \v /^ \^ /v \v /v ^\
        --
        """
        d1_1 = len(set(self.pos_out_edgelists[u]).intersection(set(self.pos_in_edgelists[v])))
        d1_2 = len(set(self.pos_out_edgelists[u]).intersection(set(self.neg_in_edgelists[v])))
        d1_3 = len(set(self.neg_out_edgelists[u]).intersection(set(self.pos_in_edgelists[v])))
        d1_4 = len(set(self.neg_out_edgelists[u]).intersection(set(self.neg_in_edgelists[v])))

        d2_1 = len(set(self.pos_out_edgelists[u]).intersection(set(self.pos_out_edgelists[v])))
        d2_2 = len(set(self.pos_out_edgelists[u]).intersection(set(self.neg_out_edgelists[v])))
        d2_3 = len(set(self.neg_out_edgelists[u]).intersection(set(self.pos_out_edgelists[v])))
        d2_4 = len(set(self.neg_out_edgelists[u]).intersection(set(self.neg_out_edgelists[v])))

        d3_1 = len(set(self.pos_in_edgelists[u]).intersection(set(self.pos_out_edgelists[v])))
        d3_2 = len(set(self.pos_in_edgelists[u]).intersection(set(self.neg_out_edgelists[v])))
        d3_3 = len(set(self.neg_in_edgelists[u]).intersection(set(self.pos_out_edgelists[v])))
        d3_4 = len(set(self.neg_in_edgelists[u]).intersection(set(self.neg_out_edgelists[v])))

        d4_1 = len(set(self.pos_in_edgelists[u]).intersection(set(self.pos_in_edgelists[v])))
        d4_2 = len(set(self.pos_in_edgelists[u]).intersection(set(self.neg_in_edgelists[v])))
        d4_3 = len(set(self.neg_in_edgelists[u]).intersection(set(self.pos_in_edgelists[v])))
        d4_4 = len(set(self.neg_in_edgelists[u]).intersection(set(self.neg_in_edgelists[v])))

        return d1_1, d1_2, d1_3, d1_4, d2_1, d2_2, d2_3, d2_4, d3_1, d3_2, d3_3, d3_4, d4_1, d4_2, d4_3, d4_4

    def get_features(self, u, v):
        x11 = self.feature_part1(u, v)
        x12 = self.feature_part2(u, v)
        return x11 + x12


def main():
    fea = FeaExtra(debug=False)
    print(fea.get_features(0, 2))
    print("test done!")

if __name__ == "__main__":
    main()
