import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
import math
import os
import argparse
from pathlib import Path
import biotite.structure.io as bsio
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam
import subprocess
import re
import ray
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler

import json
from tqdm.auto import tqdm
from ema_pytorch import EMA

from utils import PredefinedNoiseSchedule, GammaNetwork, expm1, softplus
from model.egnn_pytorch.egnn_pyg_v2 import EGNN_Sparse
from model.egnn_pytorch.utils import nodeEncoder, edgeEncoder
from dataset_src.large_dataset import Cath
from dataset_src.utils import NormalizeProtein, substitute_label
from model.egnn_pytorch.gvae_model import VGAE, Decoder
import esm
import biotite.structure.io as bsio
from Bio.PDB import PDBParser, Superimposer


def modify_suffix(lst):
    return [item + '.pt' for item in lst]

def clean_list(lst, existing_files):
    original_list = lst[:]
    lst = [item for item in lst if item in existing_files]
    removed_items = set(original_list) - set(lst)
    return lst, len(removed_items), removed_items
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Date', type=str, default='Debug',
                        help='Date of experiment')

    parser.add_argument('--dataset', type=str, default='CATH',
                        help='which dataset used for training, CATH or TS')

    parser.add_argument('--train_dir', type=str, default='dataset/cath40_k10_imem_add2ndstrc/process/',
                        help='path of training data')

    parser.add_argument('--val_dir', type=str, default='dataset/cath40_k10_imem_add2ndstrc/process/',
                        help='path of val data')

    parser.add_argument('--test_dir', type=str, default='dataset/cath40_k10_imem_add2ndstrc/process/',
                        help='path of test data')

    parser.add_argument('--ts_train_dir', type=str, default='dataset/TS/training_set/process/',
                        help='path of training data')

    parser.add_argument('--ts_test_dir', type=str, default='dataset/TS/test_set/T500/process/',
                        help='path of test data')

    parser.add_argument('--objective', type=str, default='pred_x0',
                        help='the target of training objective, objective must be either pred_x0, pred_noise or smooth_x0')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    parser.add_argument('--smooth_temperature', type=float, default=0.5,
                        help='the temperature used for smoothing label')

    parser.add_argument('--wd', type=float, default=1e-3,
                        help='weight decay')

    parser.add_argument('--drop_out', type=float, default=0.3,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--hidden_dim', type=int, default=320,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--embedding_dim', type=int, default=320,
                        help='the dim of feature embedding')

    parser.add_argument('--device_id', type=str, default="0,1,2,3",
                        help='cuda device')

    parser.add_argument('--sasa_loss_coeff', type=float, default=1.0,
                        help='the coeff of mse of sasa prediction')

    parser.add_argument('--theta', type=float, default=0.5,
                        help='the theta of ddim sample')

    parser.add_argument('--depth', type=int, default=6, # 4
                        help='number of GNN layers')

    parser.add_argument('--noise_type', type=str, default='polynomial_2', #cosine
                        help='what type of noise apply in diffusion process, uniform or blosum')

    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='clip_grad_norm')

    parser.add_argument('--pred_sasa', action='store_true', default=False,
                        help='whether predict sasa for better latent representation for mutation')

    parser.add_argument('--embedding', action='store_false',  # default = True,
                        help='whether residual embedding the feature')

    parser.add_argument('--norm_feat', action='store_false',  # default = True,
                        help='whether normalization node feature in egnn')

    parser.add_argument('--embed_ss', action='store_false',  # default = True,
                        help='whether embedding secondary structure')
    args = parser.parse_args()
    config = vars(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_id'])

    if config['dataset'] == 'CATH':
        print('train on CATH dataset')
        basedir = '../' + config['train_dir']

        with open('../dataset/chain_set_splits.json', 'r') as file:
            data = json.load(file)
        with open('../dataset/chain_set_splits.json', 'r') as file:
            test_data = json.load(file)
        # 提取三个列表
        train_ID = modify_suffix(data['train'])
        val_ID = modify_suffix(data['validation'])
        test_ID = modify_suffix(test_data['test'])



        existing_files = set(os.listdir("../dataset/cath40_k10_imem_add2ndstrc/process/"))

        # 清理每个列表并统计被删除的元素
        train_ID, train_removed_count, train_removed_items = clean_list(train_ID, existing_files)
        val_ID, validation_removed_count, validation_removed_items = clean_list(val_ID, existing_files)
        test_ID, test_removed_count, test_removed_items = clean_list(test_ID, existing_files)

        # 输出结果
        print("Train Items Removed:", train_removed_count, train_removed_items)

        print("Validation Items Removed:", validation_removed_count, validation_removed_items)

        print("Test Items Removed:", test_removed_count, test_removed_items)
        # ID = os.listdir(basedir)
        # ID.sort()
        # random.Random(4).shuffle(ID)
        # train_ID, val_ID = ID[:10000], ID[-10:]
        train_dataset = Cath(train_ID, basedir)
        val_dataset = Cath(val_ID, basedir)
        test_dataset = Cath(test_ID, basedir)


        def get_node_and_edge_counts(dataset):
            node_counts = []
            edge_counts = []
            for data in dataset:
                num_nodes = data.x.size(0)
                num_edges = data.edge_index.size(1)
                node_counts.append(num_nodes)
                edge_counts.append(num_edges)
            return node_counts, edge_counts


        def plot_statistics_and_save(node_counts, edge_counts, filename):
            # 计算平均值和中位数
            avg_node_counts = np.mean(node_counts)
            avg_edge_counts = np.mean(edge_counts)
            median_node_counts = np.median(node_counts)
            median_edge_counts = np.median(edge_counts)

            # 绘制平均节点数 vs 数据个数
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.hist(node_counts, bins=30, edgecolor='k', alpha=0.7)
            plt.axvline(avg_node_counts, color='r', linestyle='dashed', linewidth=1,
                        label=f'Average: {avg_node_counts:.2f}')
            plt.axvline(median_node_counts, color='g', linestyle='dashed', linewidth=1,
                        label=f'Median: {median_node_counts:.2f}')
            plt.xlabel('Node Count')
            plt.ylabel('Frequency')
            plt.title('Average Node Count vs. Data Count')
            plt.legend()

            # 绘制平均边数 vs 数据个数
            plt.subplot(1, 2, 2)
            plt.hist(edge_counts, bins=30, edgecolor='k', alpha=0.7)
            plt.axvline(avg_edge_counts, color='r', linestyle='dashed', linewidth=1,
                        label=f'Average: {avg_edge_counts:.2f}')
            plt.axvline(median_edge_counts, color='g', linestyle='dashed', linewidth=1,
                        label=f'Median: {median_edge_counts:.2f}')
            plt.xlabel('Edge Count')
            plt.ylabel('Frequency')
            plt.title('Average Edge Count vs. Data Count')
            plt.legend()

            plt.tight_layout()
            plt.savefig(filename)  # 保存图表到文件
            plt.show()


        # 示例代码中假设 train_dataset, val_dataset 和 test_dataset 是已经加载的数据集
        # 获取节点数和边数
        train_node_counts, train_edge_counts = get_node_and_edge_counts(train_dataset)
        val_node_counts, val_edge_counts = get_node_and_edge_counts(val_dataset)
        test_node_counts, test_edge_counts = get_node_and_edge_counts(test_dataset)

        # 汇总所有数据集中的节点数和边数
        all_node_counts = train_node_counts + val_node_counts + test_node_counts
        all_edge_counts = train_edge_counts + val_edge_counts + test_edge_counts

        # 绘制统计图
        plot_statistics_and_save(all_node_counts, all_edge_counts, 'statistics.pdf')