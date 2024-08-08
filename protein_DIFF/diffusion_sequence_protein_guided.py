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


import utils

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor < 0).any()


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def get_struc2ndRes(struc_2nds_res_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))

    if os.path.isfile(struc_2nds_res_filename):
        # open text file in read mode
        text_file = open(struc_2nds_res_filename, "r")
        # read whole file to a string
        data = text_file.read()
        # close file
        text_file.close()
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in data]
        print(len(data))
        data = F.one_hot(torch.tensor(integer_encoded), num_classes=8)
        return data
    else:
        print('Warning: ' + struc_2nds_res_filename + 'does not exist')
        return None


def pdb2graph(dataset, filename, struc_2nd_res_file):
    rec, rec_coords, c_alpha_coords, n_coords, c_coords = dataset.get_receptor_inference(filename)
    # struc_2nd_res_file = 'dataset/evaluation/DATASET/AMIE_PSEAE/ss'
    struc_2nd_res = get_struc2ndRes(struc_2nd_res_file)
    rec_graph = dataset.get_calpha_graph(
        rec, c_alpha_coords, n_coords, c_coords, rec_coords, struc_2nd_res)
    normalize_transform = NormalizeProtein(filename='dataset/cath40_k10_imem_add2ndstrc/mean_attr.pt')

    graph = normalize_transform(rec_graph)
    return graph


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.view(emb.shape[0], -1)


class EGNN_NET(torch.nn.Module):
    def __init__(self, input_feat_dim, hidden_channels, edge_attr_dim, dropout, n_layers, output_dim=320,
                 embedding=False, embedding_dim=64, update_edge=True, norm_feat=False, embedding_ss=False):
        super(EGNN_NET, self).__init__()
        torch.manual_seed(12345)
        self.dropout = dropout

        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()
        self.ff_norm_list = nn.ModuleList()
        self.sinu_pos_emb = SinusoidalPosEmb(hidden_channels)
        self.embedding = embedding
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.embedding_ss = embedding_ss

        self.time_mlp = nn.Sequential(self.sinu_pos_emb, nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
                                      nn.Linear(hidden_channels, embedding_dim))

        self.ss_mlp = nn.Sequential(nn.Linear(8, hidden_channels), nn.SiLU(),
                                    nn.Linear(hidden_channels, embedding_dim))

        for i in range(n_layers):
            if i == 0:
                layer = EGNN_Sparse(input_feat_dim, m_dim=hidden_channels, hidden_dim=hidden_channels,
                                    out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge=self.update_edge, norm_feats=norm_feat)
            else:
                layer = EGNN_Sparse(hidden_channels, m_dim=hidden_channels, hidden_dim=hidden_channels,
                                    out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge=self.update_edge, norm_feats=norm_feat)

            time_mlp_layer = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, hidden_channels * 2))
            ff_norm = torch_geometric.nn.norm.LayerNorm(hidden_channels)
            ff_layer = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 4), nn.Dropout(p=dropout), nn.GELU(),
                                     nn.Linear(hidden_channels * 4, hidden_channels))
            self.mpnn_layes.append(layer)
            self.time_mlp_list.append(time_mlp_layer)
            self.ff_list.append(ff_layer)
            self.ff_norm_list.append(ff_norm)

        self.edge_embedding = edgeEncoder(embedding_dim)
        self.lin = Linear(hidden_channels, output_dim)

    def forward(self, data, time):
        # data.x first 20 dim is noise label. 21 to 34 is knowledge from backbone,
        # e.g. mu_r_norm, sasa, b factor and so on
        x, pos, _, edge_index, edge_attr, ss, batch = data.x, data.pos, data.extra_x, data.edge_index, \
            data.edge_attr, data.ss, data.batch

        t = self.time_mlp(time)

        ss_embed = self.ss_mlp(ss)

        if self.embedding:
            edge_attr = self.edge_embedding(edge_attr)

        x = torch.cat([pos, x], dim=1)


        for i, layer in enumerate(self.mpnn_layes):
            # GNN aggregate
            if self.update_edge:
                h, edge_attr = layer(x, edge_index, edge_attr, batch)  # [N,hidden_dim]
            else:
                h = layer(x, edge_index, edge_attr, batch)  # [N,hidden_dim]

            # time and conditional shift
            corr, feats = h[:, 0:3], h[:, 3:]
            time_emb = self.time_mlp_list[i](t)  # [B,hidden_dim*2]
            scale_, shift_ = time_emb.chunk(2, dim=1)
            scale = scale_[data.batch]
            shift = shift_[data.batch]
            feats = feats * (scale + 1) + shift

            # FF neural network
            feature_norm = self.ff_norm_list[i](feats, batch)
            feats = self.ff_list[i](feature_norm) + feature_norm

            # TODO add skip connect
            x = torch.cat([corr, feats], dim=-1)

            # TODO check if only contact the ss information before last linear layer
        corr, x = x[:, 0:3], x[:, 3:]
        if self.embedding_ss:
            x = x + ss_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x, None


class Sparse_DIGRESS(nn.Module):
    def __init__(self, model, config, *, timesteps=1000, sampling_timesteps=None, loss_type='l2', objective='pred_noise',
                 beta_schedule='sigmoid', label_smooth_tem=1.0, schedule_fn_kwargs=dict(), noise_precision=1e-5,
                 Dec=None, mean_x=None, std_x=None):
        super().__init__()
        self.model = model
        # self.self_condition = self.model.self_condition
        self.objective = objective
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.noise_type = config['noise_type']
        self.config = config
        self.std = std_x
        self.mean = mean_x
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if Dec is not None:
            self.decoder = Dec.to(self.device)
        if self.noise_type == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(self.noise_type, timesteps=timesteps,
                                                 precision=noise_precision)

        self.label_smooth_tem = label_smooth_tem
        assert objective in {'pred_noise', 'pred_x0', 'pred_v',
                             'smooth_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'CE':
            return F.cross_entropy

    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def apply_noise(self, data, t_int):
        s_int = t_int - 1
        t_float = t_int / self.timesteps
        s_float = s_int / self.timesteps
        x = data.x

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.gamma(s_float)
        gamma_t = self.gamma(t_float)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)  # (32,1) (1,1)
        alpha_t_flate = alpha_t[data.batch].view(-1, 1)
        sigma_t = self.sigma(gamma_t, x)
        sigma_t_flate = sigma_t[data.batch].view(-1, 1)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = torch.randn(x.size()).type_as(x)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        noise_X = alpha_t_flate * x + sigma_t_flate * eps

        noise_data = data.clone()
        noise_data.x = noise_X.to(data.x.device)

        return noise_data, eps, sigma_t_flate, alpha_t_flate

    def sample_p_zs_given_zt(self, t, s, zt, data, Eta, last_step):
        """
        sample zs~p(zs|zt)
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        # Compute alpha_t and sigma_t from gamma.
        alpha_s = self.alpha(gamma_s, zt)  # (32,1) (1,1)
        alpha_s = alpha_s[data.batch].view(-1, 1)
        sigma_s = self.sigma(gamma_s, zt)
        sigma_s = sigma_s[data.batch].view(-1, 1)
        alpha_t = self.alpha(gamma_t, zt)  # (32,1) (1,1)
        alpha_t = alpha_t[data.batch].view(-1, 1)
        sigma_t = self.sigma(gamma_t, zt)
        sigma_t = sigma_t[data.batch].view(-1, 1)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_t_given_s = sigma_t_given_s[data.batch].view(-1, 1)
        noise_data = data.clone()
        # noise_data.x = torch.cat((zt, data.ss, data.mu_r_nor), dim=1)   # x_t


        # Neural net prediction.
        pred, _ = self.model(noise_data, t * self.timesteps)



        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        mu = alpha_s * pred + torch.sqrt(torch.pow(sigma_s, 2) - torch.pow(Eta * sigma, 2)) * (zt - alpha_t * pred) / sigma_t
        # mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * pred


        eps = torch.randn(zt.size()).type_as(zt)

        # Sample zs given the paramters derived from zt.
        zs = mu + Eta*sigma * eps

        return zs

    def sample_p_zt_1_given_zt(self, t, s, zt, data, last_step):
        """
        sample zs~p(zs|zt)
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        # Compute alpha_t and sigma_t from gamma.
        alpha_s = self.alpha(gamma_s, zt)  # (32,1) (1,1)
        alpha_s = alpha_s[data.batch].view(-1, 1)
        sigma_s = self.sigma(gamma_s, zt)
        sigma_s = sigma_s[data.batch].view(-1, 1)
        alpha_t = self.alpha(gamma_t, zt)  # (32,1) (1,1)
        alpha_t = alpha_t[data.batch].view(-1, 1)
        sigma_t = self.sigma(gamma_t, zt)
        sigma_t = sigma_t[data.batch].view(-1, 1)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        sigma2_t_given_s = sigma2_t_given_s[data.batch].view(-1, 1)
        sigma_t_given_s = sigma_t_given_s[data.batch].view(-1, 1)
        alpha_t_given_s = alpha_t_given_s[data.batch].view(-1, 1)
        noise_data = data.clone()
        noise_data.x = torch.cat((zt, data.ss, data.mu_r_nor), dim=1)

        # Neural net prediction.
        pred, _ = self.model(noise_data, t * self.timesteps)

        if last_step:
            pred = self.destandardize(pred)
            # feats, edge_index, y, coors, node_mask, edge_attr, original_x = self.decoder.to_dense(pred,
            #                                                                                       noise_data.edge_index,
            #                                                                                       noise_data.edge_attr,
            #                                                                                       noise_data.batch,
            #                                                                                       noise_data.pos,
            #                                                                                       noise_data.original_x)
            # pre_X = self.decoder(feats, coors, edge_index)
            # pre_X = pre_X[node_mask]
            pre_X = self.decoder(pred)

            sample_X = pre_X.view(-1, pre_X.shape[-1])
            return pred, sample_X
        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        mu = (alpha_t_given_s*torch.pow(sigma_s, 2)*zt+alpha_s*sigma_t_given_s*pred)/torch.pow(sigma_t, 2)

        # mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * pred


        eps = torch.randn(zt.size()).type_as(zt)

        # Sample zs given the paramters derived from zt.
        zs = mu + sigma * eps

        return zs, None

    def sample(self, data, cond=False, Eta=0.5, stop=0):
        zt = torch.randn(data.x.size()).type_as(data.x)  # [N,32]
        zt = zt.to(data.x.device)
        for s_int in tqdm(list(reversed(range(stop, self.timesteps)))):  # 500
            # z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((data.batch[-1] + 1, 1)).type_as(data.x)
            t_array = s_array + 1
            s_norm = s_array / self.timesteps
            t_norm = t_array / self.timesteps
            zt, final_predicted_X = self.sample_p_zt_1_given_zt(t_norm, s_norm, zt, data,
                                                              last_step=s_int == stop)
        return zt, final_predicted_X

    def ddim_sample(self, data, Eta=1, stop=0, step=10):
        num_samples = 6
        all_samples = []
        noisedata = torch.randn(data.x.size()).type_as(data.x)
        for i in range(num_samples):
            sample_timestep = 1000
            # noisedata, _, _, _ = self.apply_noise(data, torch.full((data.x.size(0), 1), sample_timestep))
            # zt = noisedata.x
            zt = noisedata.to(data.x.device)
            for s_int in tqdm(list(reversed(range(stop, sample_timestep - step, step)))):  # 500
                # z_t-1 ~p(z_t-1|z_t),
                s_array = s_int * torch.ones((data.batch[-1] + 1, 1)).type_as(data.x)
                t_array = s_array + step
                s_norm = s_array / self.timesteps
                t_norm = t_array / self.timesteps
                zt = self.sample_p_zs_given_zt(t_norm, s_norm, zt, data, Eta, last_step=s_int == stop)
            all_samples.append(zt)
        average_sample = torch.mean(torch.stack(all_samples), dim=0)
        zt = self.destandardize(average_sample)
        # feats, edge_index, y, coors, node_mask, edge_attr, original_x = self.decoder.to_dense(pred,
        #                                                                                       noise_data.edge_index,
        #                                                                                       noise_data.edge_attr,
        #                                                                                       noise_data.batch,
        #                                                                                       noise_data.pos,
        #                                                                                       noise_data.original_x)
        pre_X = self.decoder(zt)
        final_predicted_X = pre_X.view(-1, pre_X.shape[-1])

        return zt, final_predicted_X

    def destandardize(self, x):
        std = self.std.to(x.device)
        mean = self.mean.to(x.device)
        x = x * std + mean
        return x

    def forward(self, data, logit=False):
        data = data.to(self.device)
        if logit:
            t_int = torch.randint(0, int(self.timesteps/2), size=(data.batch[-1] + 1, 1), device=data.x.device).float()
        else:
            t_int = torch.randint(0, self.timesteps, size=(data.batch[-1] + 1, 1), device=data.x.device).float()
        noise_data, eps, sigma_t_flate, alpha_t_flate= self.apply_noise(data, t_int)
        # noise_data.x = torch.cat((noise_data.x, data.ss, data.mu_r_nor), dim=1)
        pred_X, pred_sasa = self.model(noise_data, t_int)  # have parameter

        if self.objective == 'pred_x0':
            target = data.x
        elif self.objective == 'pred_noise':
            target = eps
        elif self.objective == 'smooth_x0':
            target = substitute_label(data.x.argmax(dim=1), temperature=self.label_smooth_tem)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(pred_X, target, reduction='mean')
        if self.objective == 'pred_x0':
            pred_X = self.destandardize(pred_X)
            target = self.destandardize(target)

        elif self.objective == 'pred_noise':
            pred_X = (noise_data.x - sigma_t_flate * eps)/alpha_t_flate
            # pred_X = self.destandardize(pred_X)
            # feats, edge_index, y, coors, node_mask, edge_attr, original_x= self.decoder.to_dense(pred_X,
            #                                                                             noise_data.edge_index,
            #                                                                             noise_data.edge_attr,
            #                                                                             noise_data.batch,
            #                                                                             noise_data.pos,
            #                                                                             noise_data.original_x)


        # sample_graph= self.decoder(feats, coors, edge_index)
        # sample_graph = sample_graph[node_mask]
        sample_graph = self.decoder(pred_X)
        sample_graph = sample_graph.view(-1, sample_graph.shape[-1])
        teacher_output = self.decoder(target)
        teacher_output = teacher_output.view(-1, teacher_output.shape[-1])
        recovery = (sample_graph.argmax(dim=-1) == teacher_output.argmax(dim=-1)).sum() / data.x.shape[0]
        return loss, recovery

        # if exists(pred_sasa):
        #     mse_loss = F.mse_loss(pred_sasa, data.sasa)
        #     loss = _loss + self.config['sasa_loss_coeff'] * mse_loss
        # else:
        #     loss = _loss
        #
        # if logit:
        #     return loss, pred_X
        # else:
        #     if exists(pred_sasa):
        #         return loss, _loss, self.config['sasa_loss_coeff'] * mse_loss
        #     else:
        #         return loss, _loss, None


def seq_recovery(data, pred_seq):
    '''
    data.x is nature sequence
    '''
    recovery_list = []
    for i in range(data.ptr.shape[0] - 1):
        nature_seq = data.x[data.ptr[i]:data.ptr[i + 1], :].argmax(dim=1)
        pred = pred_seq[data.ptr[i]:data.ptr[i + 1], :].argmax(dim=1)
        recovery = (nature_seq == pred).sum() / nature_seq.shape[0]
        recovery_list.append(recovery.item())

    return recovery_list


class Trianer(object):
    def __init__(
            self,
            config,
            diffusion_model,
            train_dataset,
            val_dataset,
            test_dataset,
            *,
            train_batch_size=512,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            weight_decay=1e-2,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,  # 0.999
            adam_betas=(0.9, 0.99),
            save_and_sample_every=10000,
            num_samples=25,
            results_folder='../protein_DIFF/results',
            esmfold=None,
            esm_alphabet=None,
            amp=False,
            fp16=False,
            split_batches=True,
            convert_image_to=None
    ):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # if torch.cuda.is_available():
        #     self.model = DataParallel(diffusion_model).to(device)
        #
        # else:
        self.model = diffusion_model.to(device)

        # self.model = diffusion_model
        self.config = config
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader, num_worker 多线程

        self.ds = train_dataset
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=0)

        self.dl = cycle(dl)
        self.esmfold = esmfold
        self.esm_alphabet = esm_alphabet
        # 测试用代码
        # 创建一个专用于预提取的生成器实例
        pre_extraction_dl = cycle(dl)

        # 提取前十个数据
        self.pre_collected_data = [next(pre_extraction_dl) for _ in range(1)]

        self.val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True,
                                     num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True,
                                      num_workers=0)
        # optimizer

        self.opt = Adam(self.model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=5000, eta_min=1e-5)
        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        Path(results_folder + '/weight/').mkdir(exist_ok=True)
        Path(results_folder + '/figure/').mkdir(exist_ok=True)
        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.save_file_name = self.config[
                                  'Date'] + f"_dataset={self.config['dataset']}_result_lr={self.config['lr']}_wd={self.config['wd']}_dp={self.config['drop_out']}_hidden={self.config['hidden_dim']}_noisy_type={self.config['noise_type']}_embed_ss={self.config['embed_ss']}"



        # self.esm = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to('cuda:1')
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    def save(self, milestone):
        # if not self.accelerator.is_local_main_process:
        #     return
        # if len(self.model.device_ids) > 1:
        #     state_dict = self.model.module.state_dict()
        # else:
        state_dict = self.model.state_dict()
        data = {
            'config': self.config,
            'step': self.step,
            'model': state_dict,
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }

        torch.save(data, os.path.join(str(self.results_folder), 'weight', self.save_file_name + f'_{milestone}.pt'))

    def load(self, milestone, filename=False):
        # accelerator = self.accelerator
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if filename:
            data = torch.load(str(self.results_folder) + '/' + filename, map_location=device)
        else:
            data = torch.load(str(self.results_folder / self.config[
                'Date'] + f"model_lr={self.config['lr']}_dp={self.config['drop_out']}_timestep={self.config['timesteps']}_hidden={self.config['hidden_dim']}_{milestone}.pt"),
                              map_location=device)

        # model = self.accelerator.unwrap_model(self.model)
        # clean_dict = {}
        # for key,value in data['model'].items():
        #     clean_dict[key.replace('module.','')] = value
        # model.state_dict()[key.replace('module.','')] = value
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self, lr_config=False):
        lr_schedule = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loss, ce_loss_list, mse_loss_list, recovery_list, perplexity, val_loss_list, corr_record = [], [], [], [], [], [], []
        total_loss, total_ce_loss, total_mse_loss = 5, 5, 5
        corr_list = [[] for i in range(28)]
        val_loss = torch.tensor([5.0])

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                self.model.train()
                for _ in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)
                    data = next(self.dl)
                    if self.step < 3000:
                        loss, recovery = self.model(data, logit=True)
                    else:
                        loss, recovery = self.model(data, logit=False)
                    loss = loss / self.gradient_accumulate_every
                    # ce_loss = ce_loss / self.gradient_accumulate_every
                    total_loss += loss.mean().item()
                    # total_ce_loss += ce_loss.mean().item()
                    # if self.config['pred_sasa']:
                    #     total_mse_loss += mse_loss.mean().item()

                    loss.mean().backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                all_iter = len(self.ds) // self.batch_size
                num_iter = self.step % all_iter + 1
                pbar.set_description(f'loss: {total_loss:.4f}, recon: {recovery:.4f}')
                total_loss = 0

                if self.step % (len(self.ds) // self.batch_size) == 0 and self.step != 0:

                    train_loss.append(total_loss / all_iter)
                    ce_loss_list.append(total_ce_loss / all_iter)
                    if self.config['pred_sasa']:
                        mse_loss_list.append(total_mse_loss / all_iter)
                    val_loss_list.append(val_loss.item())
                    total_loss = 0
                    total_ce_loss = 0
                    total_mse_loss = 0

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()
                self.step += 1
                if self.step > self.train_num_steps / 2 and lr_schedule:
                    for g in self.opt.param_groups:
                        g['lr'] = self.config['lr'] * 0.1
                    lr_schedule = False
                # elif self.step <5000:#warm up
                #     for g in self.opt.param_groups:
                #         g['lr'] = 1e-7 
                # else:
                #     for g in self.opt.param_groups:
                #         g['lr'] = self.config['lr']                         

                self.ema.to(device)
                self.ema.update()

                if self.step != 0 and self.step % int(self.save_and_sample_every * (len(self.ds) // self.batch_size)) == 0:
                    self.ema.ema_model.eval()


                    with torch.no_grad():
                        # fasta_filename = "all_protein_sequences_train.fasta"
                        # # test train
                        # for data in self.pre_collected_data:
                        #     data = data.to(device)
                        #     traindata_loss, _ = self.ema.ema_model(data)
                        #     # zt,sample_graph = self.ema.ema_model.sample(data,self.config['sample_temperature'],stop = 250)
                        #     zt, sample_graph = self.ema.ema_model.ddim_sample(data, self.config['sample_temperature'],
                        #                                                       stop=0,
                        #                                                       step=50)  # zt is the output of Neural Netowrk and sample graph is a sample of it
                        #     # zt, sample_graph = self.ema.ema_model.sample(data, self.config['sample_temperature'],
                        #     #                                                   stop=0)
                        #     print(sample_graph.argmax(dim=-1))
                        #     # print(self.model.decoder(data.x).argmax(dim=-1))
                        #     print(data.original_x.argmax(dim=-1))
                        #     print(F.mse_loss(data.x, zt))
                        #     # print(traindata_loss)
                        #     recovery = (sample_graph.argmax(dim=-1) == data.original_x.argmax(dim=-1)).sum() / \
                        #                data.x.shape[0]
                        #     recovery_de = (self.model.decoder(data.x).argmax(dim=-1) == data.original_x.argmax(dim=-1)).sum() / \
                        #                     data.x.shape[0]
                        #     print(f'recovery rate is {recovery}')
                        #     print(f'decoder performance is {recovery_de}')
                        #
                        #     # # 定义氨基酸对应的编码
                        #     # amino_acid_dict = {
                        #     #     0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
                        #     #     10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y',
                        #     #     19: 'V',
                        #     #     # 这里的20-37可以根据非标准氨基酸的具体单字母表示进行调整
                        #     #     20: 'X', 21: 'X', 22: 'X', 23: 'X', 24: 'X', 25: 'X', 26: 'X', 27: 'X', 28: 'X',
                        #     #     29: 'X',
                        #     #     30: 'X', 31: 'X', 32: 'X', 33: 'X', 34: 'X', 35: 'X', 36: 'X', 37: 'X'
                        #     # }
                        #
                        #     amino_acid_dict = {i: tok for tok, i in self.esm_alphabet.tok_to_idx.items()}
                        #     unique_batches = data.batch.unique()
                        #     with open(fasta_filename, "a") as file:
                        #         for batch_id in unique_batches:
                        #             mask = data.batch == batch_id
                        #             protein_sample_graph = sample_graph[mask]
                        #             original_sample_graph = data.original_x[mask]
                        #             original_pos = data.pos[mask]
                        #
                        #             # 将one-hot编码转换为氨基酸序列
                        #             protein_sequence = ''.join(amino_acid_dict[amino_acid.item()] for amino_acid in
                        #                                        protein_sample_graph.argmax(dim=-1))
                        #             original_protein_sequence = ''.join(
                        #                 amino_acid_dict[amino_acid.item()] for amino_acid in
                        #                 original_sample_graph.argmax(dim=-1))
                        #
                        #             with torch.no_grad():
                        #                 output = self.esmfold.infer_pdb(protein_sequence)
                        #             with open("result.pdb", "w") as f:
                        #                 f.write(output)
                        #
                        #             struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
                        #             print("The train_{} pLDDT score is: {}".format(batch_id, struct.b_factor.mean()))  # this will be the pLDDT
                        #
                        #             with torch.no_grad():
                        #                 output = self.esmfold.infer_pdb(original_protein_sequence)
                        #             with open("Orignial_structure.pdb", "w") as f:
                        #                 f.write(output)
                        #
                        #             struct = bsio.load_structure("Orignial_structure.pdb", extra_fields=["b_factor"])
                        #             print("The train_{} original pLDDT score is: {}".format(batch_id, struct.b_factor.mean()))  # this will be the pLDDT
                        #
                        #             # with open('Orignial_structure.pdb', 'w') as file:
                        #             #     for i, (x, y, z) in enumerate(original_pos, start=1):
                        #             #         file.write(
                        #             #             f"ATOM  {i:5d}  CA  ALA A{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n")
                        #
                        #
                        #             command = ['TMalign', 'Orignial_structure.pdb', 'result.pdb']
                        #             result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        #                                     text=True)
                        #             print("The TM-score is: ", result.stdout)
                                    # # 将序列写入同一个FASTA文件
                                    # fasta_content = f">ProteinSequence_id{batch_id}\n{protein_sequence}\n"
                                    # file.write(fasta_content)
                                    # fasta_content = f">ProteinSequence__id{batch_id}\n{original_protein_sequence}\n"
                                    # file.write(fasta_content)
                            # for batch_id in unique_batches:
                            #     mask = data.batch == batch_id
                            #     protein_sample_graph = sample_graph[mask]
                            #
                            #     # 将one-hot编码转换为氨基酸序列
                            #     protein_sequence = ''.join(amino_acid_dict[amino_acid.item()] for amino_acid in
                            #                                protein_sample_graph.argmax(dim=-1))
                            #     with torch.no_grad():
                            #         ## output = self.esm.infer_pdb(protein_sequence)
                            #
                            #         inputs = self.tokenizer([protein_sequence], return_tensors="pt",
                            #                            add_special_tokens=False)  # A tiny random peptide
                            #         output = model(**inputs)
                            #
                            #     with open(f"result/{batch_id}_protein.pdb", "w") as f:
                            #         f.write(output)
                            #     struct = bsio.load_structure(f"result/{batch_id}_protein.pdb", extra_fields=["b_factor"])
                            #     print(f"pLDDT score of {batch_id}_protein is ", struct.b_factor.mean())  # this will be the pLDDT

                        # test eval
                        recoveries = []
                        all_prob = torch.tensor([])
                        all_seq = torch.tensor([])
                        test_plddt = []
                        original_plddt = []
                        TM_score = []
                        RMSD_score = []
                        with open("output.txt", "w") as output_file, open("input.txt", "w") as input_file, open("x_input.txt", "w") as x_input_file:
                            for batch_index, data in enumerate(self.test_loader):

                                data = data.to(device)

                                val_loss, _ = self.ema.ema_model(data)
                                # zt is the output of Neural Netowrk and sample graph is a sample of it
                                zt, sample_graph = self.ema.ema_model.ddim_sample(data, self.config['theta'],
                                                                                  stop=0,
                                                                                  step=80)
                                # zt, sample_graph = self.ema.ema_model.sample(data, self.config['theta'], stop=0)
                                print(sample_graph.argmax(dim=-1))
                                print(data.original_x.argmax(dim=-1))
                                print(F.mse_loss(data.x, zt))
                                print(val_loss)
                                print(sample_graph.argmax(dim=-1).shape)
                                print(data.original_x.argmax(dim=-1).shape)

                                recovery = (sample_graph.argmax(dim=-1) == data.original_x.argmax(dim=-1)).sum() / data.x.shape[0]
                                all_prob = torch.cat([all_prob, sample_graph.cpu()])
                                all_seq = torch.cat([all_seq, data.original_x.cpu()])

                                recoveries.append(recovery.item())
                                print(f'recovery rate is {recovery}')

                                x_to_original = self.model.decoder(self.model.destandardize(data.x))
                                x_to_original = x_to_original.argmax(dim=-1)
                                print(x_to_original)
                                recovery_de = (x_to_original == data.original_x.argmax(dim=-1)).sum() / data.x.shape[0]
                                print(f'decoder performance is {recovery_de}')
                                recovery_diffusion = (sample_graph.argmax(dim=-1) == x_to_original).sum() / data.x.shape[0]
                                print(f'diffusion performance is {recovery_diffusion}')

                                amino_acid_dict = {i: tok for tok, i in self.esm_alphabet.tok_to_idx.items()}
                                unique_batches = data.batch.unique()



                                for batch_id in unique_batches:
                                    mask = data.batch == batch_id
                                    protein_sample_graph = sample_graph[mask]
                                    original_sample_graph = data.original_x[mask]
                                    x_to_original_graph = x_to_original[mask]
                                    print(len(protein_sample_graph.argmax(dim=-1)))
                                    print(len(original_sample_graph.argmax(dim=-1)))
                                    print(len(x_to_original_graph))

                                    # 将one-hot编码转换为氨基酸序列
                                    protein_sequence = ''.join(amino_acid_dict[amino_acid.item()] for amino_acid in
                                                               protein_sample_graph.argmax(dim=-1))
                                    original_protein_sequence = ''.join(
                                        amino_acid_dict[amino_acid.item()] for amino_acid in
                                        original_sample_graph.argmax(dim=-1))
                                    x_to_original_sequence = ''.join(
                                        amino_acid_dict[amino_acid.item()] for amino_acid in
                                        x_to_original_graph)



                                    # 将序列写入同一个FASTA文件
                                    fasta_content = f"{protein_sequence}\n"
                                    output_file.write(fasta_content)
                                    fasta_content = f"{original_protein_sequence}\n"
                                    input_file.write(fasta_content)
                                    fasta_content = f"{x_to_original_sequence}\n"
                                    x_input_file.write(fasta_content)
                                    output_file.flush()
                                    input_file.flush()
                                    x_input_file.flush()


                                    # with torch.no_grad():
                                    #     output = self.esmfold.infer_pdb(protein_sequence)
                                    # with open("result.pdb", "w") as f:
                                    #     f.write(output)
                                    #
                                    # struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
                                    # print("The test_{} pLDDT score is: {}".format(batch_id,
                                    #                                                struct.b_factor.mean()))  # this will be the pLDDT
                                    # test_plddt.append(struct.b_factor.mean())
                                    #
                                    # with torch.no_grad():
                                    #     output = self.esmfold.infer_pdb(original_protein_sequence)
                                    # with open("Orignial_structure.pdb", "w") as f:
                                    #     f.write(output)
                                    #
                                    # struct = bsio.load_structure("Orignial_structure.pdb", extra_fields=["b_factor"])
                                    # print("The test_{} original pLDDT score is: {}".format(batch_id,
                                    #                                                         struct.b_factor.mean()))  # this will be the pLDDT
                                    # original_plddt.append(struct.b_factor.mean())
                                    #
                                    # # with open('Orignial_structure.pdb', 'w') as file:
                                    # #     for i, (x, y, z) in enumerate(original_pos, start=1):
                                    # #         file.write(
                                    # #             f"ATOM  {i:5d}  CA  ALA A{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n")
                                    #
                                    # command = ['TMalign', 'Orignial_structure.pdb', 'result.pdb']
                                    # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    #                         text=True)
                                    # print("The TM-score is: ", result.stdout)
                                    #
                                    # # 提取TM-score
                                    # tm_score_pattern = re.compile(r"TM-score=\s*([\d.]+)")
                                    # tm_score_match = tm_score_pattern.search(result.stdout)
                                    # if tm_score_match:
                                    #     tm_score = float(tm_score_match.group(1))
                                    #     TM_score.append(tm_score)
                                    #
                                    # # 提取RMSD
                                    # rmsd_pattern = re.compile(r"RMSD=\s*([\d.]+)")
                                    # rmsd_match = rmsd_pattern.search(result.stdout)
                                    # if rmsd_match:
                                    #     rmsd = float(rmsd_match.group(1))
                                    #     RMSD_score.append(rmsd)


                                    ##############################################
                                    # parser = PDBParser()

                                    # # 读取PDB文件
                                    # structure1 = parser.get_structure('X', 'result.pdb')
                                    # structure2 = parser.get_structure('Y', 'Orignial_structure.pdb')
                                    #
                                    #
                                    # ref_atoms = [atom for residue in structure1[0].get_residues() for atom in residue if atom.name == 'CA']
                                    # mobile_atoms = [atom for residue in structure2[0].get_residues() for atom in residue if atom.name == 'CA']
                                    # if len(ref_atoms) == len(mobile_atoms):
                                    #     # # 确保参考原子和移动原子数量一致
                                    #     # assert len(ref_atoms) == len(
                                    #     #     mobile_atoms), "{} reference atoms and {} mobile atoms must be the same."\
                                    #     #     .format(len(ref_atoms),len(mobile_atoms))
                                    #
                                    #     # 创建一个Superimposer对象
                                    #     sup = Superimposer()
                                    #     sup.set_atoms(ref_atoms, mobile_atoms)
                                    #
                                    #     # 应用最优对齐
                                    #     sup.apply(structure2.get_atoms())
                                    #
                                    #     # 打印RMSD
                                    #     print("RMSD:", sup.rms)


                        # average_test_plddt = np.mean(test_plddt)
                        # std_dev_test_plddt = np.std(test_plddt)
                        # print(
                        #     f'Average test pLDDT is {average_test_plddt:.2f}, Standard Deviation is {std_dev_test_plddt:.2f}')
                        #
                        # average_original_plddt = np.mean(original_plddt)
                        # std_dev_original_plddt = np.std(original_plddt)
                        # print(
                        #     f'Average original pLDDT is {average_original_plddt:.2f}, Standard Deviation is {std_dev_original_plddt:.2f}')
                        #
                        # if TM_score:
                        #     average_TM_score = np.mean(TM_score)
                        #     std_dev_TM_score = np.std(TM_score)
                        #     print(
                        #         f"Average TM-score is {average_TM_score:.2f}, Standard Deviation is {std_dev_TM_score:.2f}")
                        #
                        # if RMSD_score:
                        #     average_RMSD = np.mean(RMSD_score)
                        #     std_dev_RMSD = np.std(RMSD_score)
                        #     print(f"Average RMSD is {average_RMSD:.2f}, Standard Deviation is {std_dev_RMSD:.2f}")

                        recovery_list.append((sum(recoveries) / len(recoveries)))
                        print(f'recovery rate is {recovery_list[-1]}')
                        ll_fullseq = F.cross_entropy(all_prob, all_seq, reduction='mean').item()
                        perplexity.append(np.exp(ll_fullseq))
                        print(f'perplexity : {np.exp(ll_fullseq):.2f}')

                        # _,all_t_val_loss = self.ema.ema_model.compute_val_loss(data,True)

                    # weigth_corr,corr_list,DSM_list = compute_single_site_corr_score_all(self.ema.ema_model,CATH_test_inmem,corr_list,self.config['pred_sasa'])
                    # corr_record.append(weigth_corr)

                    milestone = self.step // self.save_and_sample_every
                    # Save the model
                    if recovery_list[-1] == max(recovery_list):
                        self.save(milestone)
                    break
                    # recovery_list.append(np.mean(recovery_list))
                    # print(f'recovery rate is {np.mean(recovery_list)}')
                    # perplexity.append(0.1 * np.exp(ll_fullseq))
                    # print(f'perplexity : {0.1 * np.exp(ll_fullseq):.2f}')

                    # fig = plt.figure(figsize=(8, 6))
                    # gs = GridSpec(2, 2, figure=fig)
                    #
                    # ax1 = fig.add_subplot(gs[0, 0])
                    # ax1.plot(val_loss_list, label='val_loss')
                    # ax1.plot(ce_loss_list, label='ce_loss')
                    # ax1.plot(train_loss, label='train_loss')
                    # if self.config['pred_sasa']:
                    #     ax1.plot(mse_loss_list, label='mse_loss')
                    # ax1.set_ylim((0, 4))
                    # ax1.legend(loc="upper right", fancybox=True)
                    #
                    # ax2 = fig.add_subplot(gs[1, 0])
                    # ax2.plot(recovery_list, label='recovery')
                    # ax2.plot(perplexity, label='perplexity * 0.1')
                    # ax2.legend(loc="upper right", fancybox=True)
                    # ax2.set_title(
                    #     f'best_recovery={max(recovery_list):.4f} at {recovery_list.index(max(recovery_list))}')
                    # ax2.set_ylim((0, 0.8))
                    #
                    # # ax4 = fig.add_subplot(gs[2, 0])
                    # # ax4.plot(corr_record)
                    # # ax4.set_title(f'best_corr={max(corr_record):.4f} at {corr_record.index(max(corr_record))}')
                    #
                    # # ax3 = fig.add_subplot(gs[:, 1])
                    # # for corr, protein_name in zip(corr_list, DSM_list):
                    # #     ax3.plot(corr, label=protein_name)
                    #
                    # # ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, prop={'size': 8})
                    #
                    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
                    # plt.savefig(os.path.join(str(self.results_folder), 'figure', self.save_file_name + f'.png'),
                    #             dpi=200)
                    # plt.close()

                    # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                    # self.save(milestone)

                pbar.update(1)

        print('training complete')

def standardize(data):
    data.x = (data.x - mean_x) / std_x
    return data

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

    parser.add_argument('--embed_ss', action='store_false', default = True,
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
        print(len(test_ID))
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
        # 汇总所有数据集中的x值
        all_x_values = []
        for dataset in [train_dataset, val_dataset]:
            for data in dataset:
                all_x_values.append(data.x)

        # 将列表转换为PyTorch张量
        all_x_values = torch.cat(all_x_values, dim=0)

        # 计算平均值和方差
        mean_x = all_x_values.mean(dim=0)
        std_x = all_x_values.std(dim=0)

        # 用于存储标准化后数据的新数据集
        standardized_train_dataset = []
        standardized_val_dataset = []
        standardized_test_dataset = []

        # 遍历原始数据集并应用标准化，然后将结果存储在新的数据集中
        for dataset, standardized_dataset in zip(
                [train_dataset, val_dataset, test_dataset],
                [standardized_train_dataset, standardized_val_dataset, standardized_test_dataset]
        ):
            for data in dataset:
                standardized_dataset.append(standardize(data.clone()))

        print(f'train on CATH dataset with {len(train_dataset)}  training data and {len(val_dataset)}  val data')
        del train_dataset, val_dataset, test_dataset, all_x_values # 删除原始数据集，以释放内存

    elif config['dataset'] == 'TS':
        basedir = config['train_dir']
        train_ID, val_ID = os.listdir(config['ts_train_dir']), os.listdir(config['ts_test_dir'])
        train_dataset = Cath(train_ID, config['ts_train_dir'])
        val_dataset = Cath(val_ID, config['ts_test_dir'])
        test_dataset = Cath(val_ID, config['ts_test_dir'])
        print(f'train on TS dataset with {len(train_dataset)}  training data and {len(val_dataset)}  val data')
    else:
        raise ValueError(f"unknown dataset")

    input_feat_dim = standardized_train_dataset[0].x.shape[1]
    edge_attr_dim = standardized_train_dataset[0].edge_attr.shape[1]
    # checkpoint = torch.load('protein_DIFF/results/weight/May_11th_result_lr=0.0005_wd=5e-05_dp=0.08_hidden=128_noisy_type=blosum_embed_ss=True_20619.pt',map_location=torch.device('cuda:0'))
    # config = checkpoint['config']
    # config['dataset'] = 'TS'
    if config['pred_sasa']:
        model = EGNN_NET(input_feat_dim=input_feat_dim, hidden_channels=config['hidden_dim'],
                         edge_attr_dim=edge_attr_dim, dropout=config['drop_out'], n_layers=config['depth'],
                         update_edge=True, embedding=config['embedding'], embedding_dim=config['embedding_dim'],
                         norm_feat=config['norm_feat'], output_dim=21, embedding_ss=config['embed_ss'])
    else:
        model = EGNN_NET(input_feat_dim=input_feat_dim, hidden_channels=config['hidden_dim'],
                         edge_attr_dim=edge_attr_dim, dropout=config['drop_out'], n_layers=config['depth'],
                         update_edge=True, embedding=config['embedding'], embedding_dim=config['embedding_dim'],
                         norm_feat=config['norm_feat'], embedding_ss=config['embed_ss'])  # GVP,Protein_MPNN,

    # model = DataParallel(model)

    # vgae = VGAE(39, 20, 64, 64, 93)
    # vgae.load_state_dict(torch.load('./VGAE/gae.pt'))
    # Dec = Decoder(1293, 20, 1293, 93)
    # Dec.load_state_dict(torch.load('./VGAE/gae.pt'))

    # # 设置新的路径
    # new_path = "/hy-tmp/conda/torch"
    # # 确保新的目录存在，如果不存在则创建
    # os.makedirs(new_path, exist_ok=True)
    #
    # # 设置环境变量
    # os.environ["TORCH_HOME"] = new_path
    # Load ESM-2 model
    esmfold_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    decoder = esmfold_model.lm_head
    del esmfold_model
    batch_converter = alphabet.get_batch_converter()
    esmfold= esm.pretrained.esmfold_v1()
    esmfold = esmfold.eval().cuda()


    diffusion = Sparse_DIGRESS(model=model, config=config, timesteps=config['timesteps'], objective=config['objective'],
                               label_smooth_tem=config['smooth_temperature'], Dec=decoder,
                               mean_x=mean_x, std_x=std_x)


    trainer = Trianer(config,
                      diffusion,
                      standardized_train_dataset,
                      standardized_val_dataset,
                      standardized_test_dataset,
                      train_batch_size=8,
                      gradient_accumulate_every=4,#4
                      save_and_sample_every=2,#2
                      train_num_steps=20000,
                      train_lr=config['lr'],
                      esmfold=esmfold,
                      esm_alphabet=alphabet,)
    trainer.train()


    # # 初始化 Ray
    # ray.init()
    #
    # # 配置搜索空间
    # lr_config = {
    #     'lr': tune.loguniform(1e-5, 1e-1)  # 使用对数均匀分布搜索学习率
    # }
    # # 启动超参数搜索
    # analysis = tune.Tuner(
    #     train_diffusion,
    #     tune_config=tune.TuneConfig(
    #         metric="plddt",
    #         mode="max",
    #         num_samples=1,
    #     ),
    #     param_space=lr_config,
    # )
    # analysis.fit()
    # #获取最佳配置和结果
    # best_config = analysis.get_best_config(metric="plddt", mode="max")
    # print("最佳学习率配置:", best_config)
    # ray.shutdown()