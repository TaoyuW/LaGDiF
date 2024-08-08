import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset_src import utils
from egnn_pytorch import EGNN
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch_geometric
from model.egnn_pytorch.utils import nodeEncoder, edgeEncoder
import model.egnn_pytorch.egnn_pyg_v2 as EGNN_Sparse
import esm

class EGNN_NET(torch.nn.Module):
    def __init__(self, input_feat_dim, hidden_channels, edge_attr_dim, dropout, n_layers, output_dim=20,
                 embedding=False, embedding_dim=64, update_edge=True, norm_feat=False, embedding_ss=False):
        super(EGNN_NET, self).__init__()
        torch.manual_seed(12345)
        self.dropout = dropout

        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()
        self.ff_norm_list = nn.ModuleList()
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
                layer = EGNN_Sparse(embedding_dim, m_dim=hidden_channels, hidden_dim=hidden_channels,
                                    out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge=self.update_edge, norm_feats=norm_feat)
            else:
                layer = EGNN_Sparse(hidden_channels, m_dim=hidden_channels, hidden_dim=hidden_channels,
                                    out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge=self.update_edge, norm_feats=norm_feat)

            ff_norm = torch_geometric.nn.norm.LayerNorm(hidden_channels)
            ff_layer = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 4), nn.Dropout(p=dropout), nn.GELU(),
                                     nn.Linear(hidden_channels * 4, hidden_channels))

            self.mpnn_layes.append(layer)
            self.ff_list.append(ff_layer)
            self.ff_norm_list.append(ff_norm)

        if output_dim == 20:
            self.node_embedding = nodeEncoder(embedding_dim, feature_num=4)
        else:
            self.node_embedding = nodeEncoder(embedding_dim, feature_num=3)

        self.edge_embedding = edgeEncoder(embedding_dim)
        self.lin = nn.Linear(hidden_channels, output_dim)

    def forward(self, data, time):
        # data.x first 20 dim is noise label. 21 to 34 is knowledge from backbone, e.g. mu_r_norm, sasa, b factor and so on
        x, pos, extra_x, edge_index, edge_attr, ss, batch = data.x, data.pos, data.extra_x, data.edge_index, data.edge_attr, data.ss, data.batch

        t = self.time_mlp(time)

        ss_embed = self.ss_mlp(ss)

        x = torch.cat([x, extra_x], dim=1)
        if self.embedding:
            x = self.node_embedding(x)
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

        corr, x = x[:, 0:3], x[:, 3:]
        if self.embedding_ss:
            x = x + ss_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        if self.output_dim == 21:
            return x[:, :20], x[:, 20]
        else:
            return x, None





class VGAE(torch.nn.Module):
    def __init__(self, num_nodes_feature, output_feature, hidden_channels, latent_channels, edge_dim, device):
        super(VGAE, self).__init__()
        self.device = device
        self.emb = torch.nn.Linear(num_nodes_feature, hidden_channels)
        self.egnn1 = EGNN(dim=hidden_channels, edge_dim=edge_dim, m_dim=hidden_channels,norm_feats=True, norm_coors=True)
        self.egnn2 = EGNN(dim=hidden_channels, edge_dim=edge_dim, m_dim=hidden_channels,norm_feats=True, norm_coors=True)
        self.egnn3 = EGNN(dim=hidden_channels, edge_dim=edge_dim, m_dim=hidden_channels,norm_feats=True, norm_coors=True)
        self.egnn4 = EGNN(dim=hidden_channels, edge_dim=edge_dim, m_dim=hidden_channels,norm_feats=True, norm_coors=True)
        self.fc1 = nn.Linear(hidden_channels, latent_channels)
        self.fc2 = nn.Linear(hidden_channels, latent_channels)
        self.egnn5 = EGNN(dim=latent_channels, edge_dim=edge_dim, m_dim=hidden_channels)
        self.fc3 = nn.Linear(latent_channels, output_feature)
        self.dropout = nn.Dropout(p=0.3)


    def encode(self, feats, coors, edge_index):
        feats = self.emb(feats)
        feats, coors = self.egnn1(feats, coors, edge_index)
        feats, coors = self.egnn2(feats, coors, edge_index)
        feats, coors = self.egnn3(feats, coors, edge_index)
        mu = self.fc1(feats)
        logvar = self.fc2(feats)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, feats, coors, edge_index):
        # _, edge_index, coors = data.x, data.edge_index, data.pos
        feats = self.fc3(feats)
        return feats, coors

    def forward(self, feats, coors, edge_index):
        mu, logvar = self.encode(feats, coors, edge_index)
        z = self.reparameterize(mu, logvar)
        x_hat, _ = self.decode(z, coors, edge_index)
        return x_hat, mu, logvar

    def loss_function(self, x_hat_logits, x, mu, logvar, node_mask):
        # Reshape x_hat_logits and x_p_indices as before
        x_AA = x[..., :20].argmax(dim=-1)
        x_hat_logits = x_hat_logits.view(-1, 20)  # (batch_size * node_num, num_classes)
        x_AA = x_AA.view(-1)  # (batch_size * node_num)

        # Reshape node_mask and create a mask for the valid entries
        flat_node_mask = node_mask.view(-1)  # (batch_size * node_num)

        # Select only the entries where node_mask is True
        selected_x_hat_logits = x_hat_logits[flat_node_mask]
        selected_x_AA = x_AA[flat_node_mask]

        # Compute reconstruction loss only on the masked entries
        recon_loss_protein = F.cross_entropy(selected_x_hat_logits, selected_x_AA)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss_protein + 1.0e-06*kl_loss

    def train_(self, model, data_loader, optimizer, N, device):
        model.train()
        total_loss = 0
        accumulation_steps = 0
        for data in data_loader:
            data = data.to(device)
            feats, edge_index, y, coors, node_mask, edge_attr = self.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.pos)
            recon, mu, logvar = model(feats, coors, edge_index)
            loss = self.loss_function(recon, feats, mu, logvar, node_mask)
            # 损失标量化，根据累积步数缩放损失，避免梯度过大
            (loss / N).backward()
            total_loss += loss.item()

            accumulation_steps += 1  # 累积一步梯度

            # 如果累积到了设定的步数，则进行一次参数更新
            if accumulation_steps % N == 0:
                clip_grad_norm_(model.parameters(), 1, norm_type=2.0)
                optimizer.step()  # 更新模型参数
                optimizer.zero_grad()  # 清空累积的梯度

        return total_loss / len(data_loader)

    def evaluate_(self, model, data_loader, device):
        model.eval()
        total_mismatch = 0
        total_nodes = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                feats, edge_index, y, coors, node_mask, edge_attr = self.to_dense(data.x, data.edge_index,
                                                                                   data.edge_attr,
                                                                                   data.batch, data.pos)
                recon_batch, mu, logvar = model(feats, coors, edge_index)
                # don't consider the last 6 features
                # recon_batch = recon_batch[:, :-6]
                # data_x_sliced = data.x[:, :, :-6]
                data_x_sliced = feats[..., :20]

                # Convert logits to one-hot vectors
                recon_onehot = torch.zeros_like(recon_batch)
                recon_onehot.scatter_(-1, recon_batch.argmax(dim=-1, keepdim=True), 1)

                # Check if the reconstructed one-hot vector matches the original one
                matches = (recon_onehot.argmax(dim=-1) == data_x_sliced.argmax(dim=-1)).float()
                # mask the nodes that are not in the protein
                masked_matches = matches * node_mask.float()
                correct_matches = masked_matches.sum().item()
                mismatch = node_mask.float().sum().item() - correct_matches
                total_mismatch += mismatch
                total_nodes += data.x.size(0)

        mismatch_percentage = (total_mismatch / total_nodes) * 100
        return mismatch_percentage

    def encode_no_edge(self, E):
        assert len(E.shape) == 4
        if E.shape[-1] == 0:
            return E
        no_edge = torch.sum(E, dim=3) == 0
        first_elt = E[:, :, :, 0]
        first_elt[no_edge] = 1
        E[:, :, :, 0] = first_elt
        diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
        E[diag] = 0
        return E

    def to_dense(self, x, edge_index, edge_attr, batch, pos):
        X, node_mask = to_dense_batch(x=x, batch=batch)
        Pos, _ = to_dense_batch(x=pos, batch=batch)
        # 计算均值和标准差，注意dim=1是沿着node_num的维度
        mean_pos = Pos.mean(dim=1, keepdim=True)
        std_pos = Pos.std(dim=1, keepdim=True)

        # 进行标准化
        norm_Pos = (Pos - mean_pos) / std_pos
        # node_mask = node_mask.float()

        edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
        max_num_nodes = X.size(1)
        E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
        E = self.encode_no_edge(E)
        y = None
        return X, E, y, norm_Pos, node_mask, edge_attr




class GAE(torch.nn.Module):
    def __init__(self, num_nodes_feature, output_feature, hidden_dim, latent_dim, edge_dim, device):
        super(GAE, self).__init__()
        self.device = device

        self.egnn1 = EGNN(dim=num_nodes_feature, edge_dim=edge_dim, m_dim=hidden_dim)
        self.fc = nn.Linear(num_nodes_feature, latent_dim)
        self.egnn4 = EGNN(dim=latent_dim, edge_dim=edge_dim, m_dim=hidden_dim)
        self.fc1 = nn.Linear(latent_dim, output_feature)
        # self.bn1 = nn.BatchNorm1d(latent_dim)

        # # Encoder layers
        # self.fc_edge = nn.Linear(93, 1)
        # self.conv1 = GCNConv(num_nodes_feature, latent_channels)
        # self.bn1 = nn.BatchNorm1d(latent_channels)

        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.bn2 = nn.BatchNorm1d(hidden_channels)
        #
        # self.conv3 = GCNConv(hidden_channels, latent_channels)
        # self.bn3 = nn.BatchNorm1d(latent_channels)

        # # Decoder layers
        # self.conv4 = GCNConv(latent_channels, hidden_channels)
        # self.bn4 = nn.BatchNorm1d(hidden_channels)

        # self.conv5 = GCNConv(hidden_channels, hidden_channels)
        # self.bn5 = nn.BatchNorm1d(hidden_channels)
        #
        # self.conv6 = GCNConv(latent_channels, hidden_channels)
        # self.bn6 = nn.BatchNorm1d(hidden_channels)
        #
        # self.fc = nn.Linear(hidden_channels, output_feature)
        # self.dropout = nn.Dropout(p=0.5)
        # self.relu = nn.ReLU()

    def encode(self, feats, coors, edge_index):
        feats, coors = self.egnn1(feats, coors, edge_index)
        # feats, coors = self.egnn2(feats, coors, edge_index)
        # feats, coors = self.egnn3(feats, coors, edge_index)
        feats = self.fc(feats)
        # batch_size, node_number, feature_dim = feats.shape
        # feats = feats.view(-1, feature_dim)  # reshape to (Batch * node number, feature dim)
        # feats = self.bn1(feats)
        # feats = feats.view(batch_size, node_number, feature_dim)  # reshape back to original shap
        return feats, coors

        # x = torch.cat([pos, x], dim=-1)
        # x = self.conv1(x, edge_index)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        # x = self.conv2(x, edge_index)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        #
        # x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = self.relu(x)
        # return x

    def decode(self, x, coors, edge_index):
        # _, edge_index, coors = data.x, data.edge_index, data.pos

        feats, coors = self.egnn4(x, coors, edge_index)
        feats = self.fc1(feats)
        return feats, coors
        # edge_index = data.edge_index
        # x = self.conv4(x, edge_index)
        # x = self.bn4(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        #
        # x = self.conv5(x, edge_index)
        # x = self.bn5(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        # x = self.conv6(x, edge_index)
        # x = self.bn6(x)
        # x = self.relu(x)
        #
        # x = self.fc(x)
        # return x

    def forward(self, feats, coors, edge_index):
        x_encoded, coors = self.encode(feats, coors, edge_index)
        x_hat, _ = self.decode(x_encoded, coors, edge_index)
        return x_hat

    def loss_function(self, x_hat_logits, x, node_mask):
        # Reshape x_hat_logits and x_p_indices as before
        x_p_indices = x[..., :-6].argmax(dim=-1)
        x_hat_logits = x_hat_logits.view(-1, 20)  # (batch_size * node_num, num_classes)
        x_p_indices = x_p_indices.view(-1)  # (batch_size * node_num)

        # Reshape node_mask and create a mask for the valid entries
        flat_node_mask = node_mask.view(-1)  # (batch_size * node_num)

        # Select only the entries where node_mask is True
        selected_x_hat_logits = x_hat_logits[flat_node_mask]
        selected_x_p_indices = x_p_indices[flat_node_mask]

        # Compute reconstruction loss only on the masked entries
        recon_loss_protein = F.cross_entropy(selected_x_hat_logits, selected_x_p_indices)

        return recon_loss_protein

    def train_(self, model, data_loader, optimizer, N,device):
        model.train()
        total_loss = 0
        accumulation_steps = 0
        for data in data_loader:
            data = data.to(device)
            feats, edge_index, y, coors, node_mask, edge_attr = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.pos)
            recon_batch = model(feats, coors, edge_index)
            loss = self.loss_function(recon_batch, feats, node_mask)

            # 损失标量化，根据累积步数缩放损失，避免梯度过大
            (loss / N).backward()
            total_loss += loss.item()

            accumulation_steps += 1  # 累积一步梯度

            # 如果累积到了设定的步数，则进行一次参数更新
            if accumulation_steps % N == 0:
                clip_grad_norm_(model.parameters(), 1, norm_type=2.0)
                optimizer.step()  # 更新模型参数
                optimizer.zero_grad()  # 清空累积的梯度
        return total_loss / len(data_loader)

    def evaluate_(self, model, data_loader, device):
        model.eval()
        total_mismatch = 0
        total_nodes = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                feats, edge_index, y, coors, node_mask, edge_attr = utils.to_dense(data.x, data.edge_index,
                                                                                   data.edge_attr,
                                                                                   data.batch, data.pos)
                recon_batch = model(feats, coors, edge_index)
                # don't consider the last 6 features
                # recon_batch = recon_batch[:, :-6]
                # data_x_sliced = data.x[:, :, :-6]
                data_x_sliced = feats[..., :-6]

                # Convert logits to one-hot vectors
                recon_onehot = torch.zeros_like(recon_batch)
                recon_onehot.scatter_(-1, recon_batch.argmax(dim=-1, keepdim=True), 1)

                # Check if the reconstructed one-hot vector matches the original one
                matches = (recon_onehot.argmax(dim=-1) == data_x_sliced.argmax(dim=-1)).float()
                # mask the nodes that are not in the protein
                masked_matches = matches * node_mask.float()
                correct_matches = masked_matches.sum().item()
                mismatch = node_mask.float().sum().item() - correct_matches
                total_mismatch += mismatch
                total_nodes += data.x.size(0)

        mismatch_percentage = (total_mismatch / total_nodes) * 100
        return mismatch_percentage


class Decoder(torch.nn.Module):
    def __init__(self, num_nodes_feature, output_feature, hidden_channels, edge_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_feature)
        self.silu1 = nn.SiLU()
        self.egnn1 = EGNN(dim=num_nodes_feature, edge_dim=edge_dim, m_dim=hidden_channels)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, feats, coors, edge_index):
        # feats, coors = self.egnn1(feats, coors, edge_index)
        feats = self.fc1(feats)
        feats = self.silu1(feats)
        feats = self.fc2(feats)
        return feats

    def loss_function(self, x_hat_logits, x, node_mask):
        # Reshape x_hat_logits and x_p_indices as before
        x_AA = x[..., :20].argmax(dim=-1)
        x_hat_logits = x_hat_logits.view(-1, 20)  # (batch_size * node_num, num_classes)
        x_AA = x_AA.view(-1)  # (batch_size * node_num)

        # Reshape node_mask and create a mask for the valid entries
        flat_node_mask = node_mask.view(-1)  # (batch_size * node_num)

        # Select only the entries where node_mask is True
        x_hat_logits = x_hat_logits[flat_node_mask]
        x_AA = x_AA[flat_node_mask]

        # Compute reconstruction loss only on the masked entries
        recon_loss_protein = F.cross_entropy(x_hat_logits, x_AA)

        return recon_loss_protein

    def train_(self, model, data_loader, optimizer, N, device):
        model.train()
        total_loss = 0
        accumulation_steps = 0
        for data in data_loader:
            data = data.to(device)
            feats, edge_index, y, coors, node_mask, edge_attr, original_x = self.to_dense(data.x, data.edge_index,
                                                                                          data.edge_attr, data.batch,
                                                                                          data.pos, data.original_x)
            feats = model(feats, coors, edge_index)
            loss = self.loss_function(feats, original_x, node_mask)
            # 损失标量化，根据累积步数缩放损失，避免梯度过大
            (loss / N).backward()
            total_loss += loss.item()

            accumulation_steps += 1  # 累积一步梯度

            # 如果累积到了设定的步数，则进行一次参数更新
            if accumulation_steps % N == 0:
                clip_grad_norm_(model.parameters(), 1, norm_type=2.0)
                optimizer.step()  # 更新模型参数
                optimizer.zero_grad()  # 清空累积的梯度

        return total_loss / len(data_loader)

    def evaluate_(self, model, data_loader, device):
        model.eval()
        total_mismatch = 0
        total_nodes = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                feats, edge_index, y, coors, node_mask, edge_attr, original_x = self.to_dense(data.x, data.edge_index,
                                                                                              data.edge_attr,
                                                                                              data.batch,
                                                                                              data.pos, data.original_x)
                recon_batch = model(feats, coors, edge_index)
                # don't consider the last 6 features
                # recon_batch = recon_batch[:, :-6]
                # data_x_sliced = data.x[:, :, :-6]
                data_x_sliced = original_x[..., :20]

                # Convert logits to one-hot vectors
                recon_onehot = torch.zeros_like(recon_batch)
                recon_onehot.scatter_(-1, recon_batch.argmax(dim=-1, keepdim=True), 1)

                # Check if the reconstructed one-hot vector matches the original one
                matches = (recon_onehot.argmax(dim=-1) == data_x_sliced.argmax(dim=-1)).float()
                # mask the nodes that are not in the protein
                masked_matches = matches * node_mask.float()
                correct_matches = masked_matches.sum().item()
                mismatch = node_mask.float().sum().item() - correct_matches
                total_mismatch += mismatch
                total_nodes += data.x.size(0)

        mismatch_percentage = (total_mismatch / total_nodes) * 100
        return mismatch_percentage

    def encode_no_edge(self, E):
        assert len(E.shape) == 4
        if E.shape[-1] == 0:
            return E
        no_edge = torch.sum(E, dim=3) == 0
        first_elt = E[:, :, :, 0]
        first_elt[no_edge] = 1
        E[:, :, :, 0] = first_elt
        diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
        E[diag] = 0
        return E

    def to_dense(self, x, edge_index, edge_attr, batch, pos, original_x):
        X, node_mask = to_dense_batch(x=x, batch=batch)
        original_x, _ = to_dense_batch(x=original_x, batch=batch)
        Pos, _ = to_dense_batch(x=pos, batch=batch)
        # 计算均值和标准差，注意dim=1是沿着node_num的维度
        mean_pos = Pos.mean(dim=1, keepdim=True)
        std_pos = Pos.std(dim=1, keepdim=True)

        # 进行标准化
        norm_Pos = (Pos - mean_pos) / std_pos
        # node_mask = node_mask.float()

        edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
        max_num_nodes = X.size(1)
        E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
        E = self.encode_no_edge(E)
        y = None
        return X, E, y, norm_Pos, node_mask, edge_attr, original_x