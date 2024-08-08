import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import utils
from egnn_pytorch import EGNN
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import GraphNorm
from model import egnn_pytorch

class VGAE(torch.nn.Module):
    def __init__(self, num_nodes_feature, output_feature, hidden_channels, latent_channels, edge_dim, device):
        super(VGAE, self).__init__()
        self.device = device
        self.egnn1 = EGNN(dim=num_nodes_feature, edge_dim=edge_dim, m_dim=hidden_channels)
        self.egnn2 = EGNN(dim=num_nodes_feature, edge_dim=edge_dim, m_dim=hidden_channels)
        self.egnn3 = EGNN(dim=num_nodes_feature, edge_dim=edge_dim, m_dim=hidden_channels)
        self.fc1 = nn.Linear(num_nodes_feature, latent_channels)
        self.fc2 = nn.Linear(num_nodes_feature, latent_channels)
        self.egnn4 = EGNN(dim=latent_channels, edge_dim=edge_dim, m_dim=hidden_channels)
        self.fc3 = nn.Linear(latent_channels, output_feature)
        self.bn1 = nn.BatchNorm1d(num_nodes_feature)
        self.dropout = nn.Dropout(p=0.5)

    def encode(self, feats, coors, edge_index):
        feats, coors = self.egnn1(feats, coors, edge_index)
        # feats, coors = self.egnn2(feats, coors, edge_index)
        # feats, coors = self.egnn3(feats, coors, edge_index)
        mu = self.fc1(feats)
        logvar = self.fc2(feats)
        return mu, logvar, coors

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, coors, edge_index):
        # _, edge_index, coors = data.x, data.edge_index, data.pos

        feats, coors = self.egnn4(x, coors, edge_index)
        feats = self.fc3(feats)
        return feats, coors

    def forward(self, feats, coors, edge_index):
        mu, logvar, coors = self.encode(feats, coors, edge_index)
        z = self.reparameterize(mu, logvar)
        x_hat, _ = self.decode(z, coors, edge_index)
        return x_hat, mu, logvar

    def loss_function(self, x_hat_logits, x, mu, logvar, node_mask):
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

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss_protein + 1.0e-06*kl_loss

    def train_(self, model, data_loader, optimizer, device):
        model.train()
        total_loss = 0
        for data in data_loader:
            data = data.to(device)
            feats, edge_index, y, coors, node_mask, edge_attr = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.pos)
            recon, mu, logvar = model(feats, coors, edge_index)
            loss = self.loss_function(recon,  feats, mu, logvar, node_mask)

            # loss = loss_function(recon_batch, data.x)
            clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            optimizer.zero_grad()
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
                recon_batch, mu, logvar = model(feats, coors, edge_index)
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

    def train_(self, model, data_loader, optimizer, device):
        model.train()
        total_loss = 0
        for data in data_loader:
            data = data.to(device)
            feats, edge_index, y, coors, node_mask, edge_attr = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.pos)
            recon_batch = model(feats, coors, edge_index)
            loss = self.loss_function(recon_batch, feats, node_mask)

            # loss = loss_function(recon_batch, data.x)
            clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            optimizer.zero_grad()
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