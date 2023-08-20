import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
from model.egnn_pytorch import EGNN_Network_ as eg
from dgd.models.transformer_model import GraphTransformer
from dgd.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition, PredefinedNoiseSchedule
from dgd.diffusion import diffusion_utils
from dgd.metrics.train_metrics import TrainLoss_Protein
from dgd.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL, SumExceptBatchMSE
from torch.nn.utils import clip_grad_norm_
from dgd import utils
import GPUtil





class DiscreteDenoisingDiffusion_protein(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        # nodes_dist = dataset_infos.nodes_dist
        self.norm_values = cfg.model.normalize_factors
        self.norm_biases = cfg.model.norm_biases

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps
        self.generate_E = cfg.dataset.generate_E
        self.generate_y = cfg.dataset.generate_y

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        # self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLoss_Protein(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_mse = SumExceptBatchMSE()
        self.val_E_mse = SumExceptBatchMSE()
        self.val_y_mse = SumExceptBatchMSE()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMSE()

        self.test_nll = NLL()
        self.test_X_mse = SumExceptBatchMSE()
        self.test_E_mse = SumExceptBatchMSE()
        self.test_y_mse = SumExceptBatchMSE()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMSE()

        # self.train_metrics = train_metrics
        # self.sampling_metrics = sampling_metrics

        # self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        if self.cfg.model.model == 'graph_tf':
            self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                        input_dims=input_dims,
                                        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                        hidden_dims=cfg.model.hidden_dims,
                                        output_dims=output_dims,
                                        act_fn_in=nn.ReLU(),
                                        act_fn_out=nn.ReLU(),
                                        update_edge = self.generate_E,
                                        update_y = self.generate_y)

        elif self.cfg.model.model == 'EGNN':
            self.model = eg(input_dim=input_dims["X"], output_dim=16, edge_dim=input_dims["E"], depth=5)

        elif self.cfg.model.model == 'GVP':
            pass
        elif self.cfg.model.model == 'GVP_tf':
            pass
        elif self.cfg.model.model == 'Protein_MPNN':
            pass

        self.gamma = PredefinedNoiseSchedule(cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        # self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.loss_type = cfg.train.loss_type
        # self.Tracker = tracker.SummaryTracker()

    def training_step(self, data, i):

        dense_data, node_mask, dense_extra_X, edge_attr = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.extra_x, data.pos)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data, dense_extra_X)
        pred = self.forward(dense_data.pos, noisy_data, extra_data, node_mask, edge_attr)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,noise=noisy_data["epsX"],
                               loss_type=self.loss_type,log=i % self.log_every_steps == 0)
        # self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
        #                    log=i % self.log_every_steps == 0)
        if torch.isinf(loss) or torch.isnan(loss):
            print("1")
        print(loss)
        return {'loss': loss}

    def optimizer_step(self,epoch,batch_idx,optimizer,optimizer_closure,):
        # 梯度截断
        clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        # self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time)
        # self.train_metrics.log_epoch_metrics(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_mse.reset()
        self.val_E_mse.reset()
        self.val_y_mse.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_logp.reset()
        # self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask,dense_extra_X, edge_attr = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch,data.extra_x, data.pos)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data, dense_extra_X)
        pred = self.forward(dense_data.pos, noisy_data, extra_data, node_mask, edge_attr)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, dense_extra_X, node_mask, dense_data.pos, edge_attr, test=False )
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_mse.compute(), self.val_E_mse.compute(),
                   self.val_y_mse.compute(), self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_y_logp.compute()]
        wandb.log({"val/epoch_NLL": metrics[0],
                   "val/X_kl": metrics[1],
                   "val/E_kl": metrics[2],
                   "val/y_kl": metrics[3],
                   "val/X_logp": metrics[4],
                   "val/E_logp": metrics[5],
                   "val/y_logp": metrics[6]}, commit=False)

        print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val amino acid type KL {metrics[1] :.2f} -- ",
              f"Val Edge type KL: {metrics[2] :.2f} -- Val Global feat. KL {metrics[3] :.2f}\n")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            print("Computing sampling metrics...")
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_mse.reset()
        self.test_E_mse.reset()
        self.test_y_mse.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_logp.reset()

    def test_step(self, data, i):
        dense_data, node_mask,dense_extra_X = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch,data.extra_x)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data, dense_extra_X)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, dense_extra_X,node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_mse.compute(), self.test_E_mse.compute(),
                   self.test_y_mse.compute(), self.test_X_logp.compute(), self.test_E_logp.compute(),
                   self.test_y_logp.compute()]
        wandb.log({"test/epoch_NLL": metrics[0],
                   "test/X_mse": metrics[1],
                   "test/E_mse": metrics[2],
                   "test/y_mse": metrics[3],
                   "test/X_logp": metrics[4],
                   "test/E_logp": metrics[5],
                   "test/y_logp": metrics[6]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
              f"Test Edge type KL: {metrics[2] :.2f} -- Test Global feat. KL {metrics[3] :.2f}\n")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        print("Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()
        print("Done.")


    def reconstruction_logp(self, data, data_0, gamma_0, eps, pred_0, node_mask, epsilon=1e-10, test=False):
        """ Reconstruction loss.
            output size: (1).
        """
        X, E, y = data.values()
        X_0, E_0, y_0 = data_0.values()

        # TODO: why don't we need the values of X and E?
        _, _, eps_y0 = eps.values()
        predy = pred_0.y

        # 2. Compute reconstruction loss for integer/categorical features on nodes and edges

        # Compute sigma_0 and rescale to the integer scale of the data_utils.
        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=X_0.size())
        sigma_0_X = sigma_0 * self.norm_values[0]
        sigma_0_E = (sigma_0 * self.norm_values[1]).unsqueeze(-1)

        # Unnormalize features
        unnormalized_data = utils.unnormalize(X, E, y, self.norm_values, self.norm_biases, node_mask, collapse=False)
        unnormalized_0 = utils.unnormalize(X_0, E_0, y_0, self.norm_values, self.norm_biases, node_mask, collapse=False)
        X_0, E_0, _ = unnormalized_0.X, unnormalized_0.E, unnormalized_0.y
        assert unnormalized_data.X.size() == X_0.size()

        # Centered cat features around 1, since onehot encoded.
        E_0_centered = E_0 - 1
        X_0_centered = X_0 - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        log_pE_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((E_0_centered + 0.5) / sigma_0_E)
            - diffusion_utils.cdf_std_gaussian((E_0_centered - 0.5) / sigma_0_E)
            + epsilon)

        log_pX_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((X_0_centered + 0.5) / sigma_0_X)
            - diffusion_utils.cdf_std_gaussian((X_0_centered - 0.5) / sigma_0_X)
            + epsilon)

        # Normalize the distributions over the categories.
        norm_cst_E = torch.logsumexp(log_pE_proportional, dim=-1, keepdim=True)
        norm_cst_X = torch.logsumexp(log_pX_proportional, dim=-1, keepdim=True)

        log_probabilities_E = log_pE_proportional - norm_cst_E
        log_probabilities_X = log_pX_proportional - norm_cst_X

        # Select the log_prob of the current category using the one-hot representation.
        logps = utils.PlaceHolder(X=log_probabilities_X * unnormalized_data.X,
                                  E=log_probabilities_E * unnormalized_data.E,
                                  y=None).mask(node_mask)

        if test:
            log_pX = - self.test_X_logp(-logps.X)
        else:
            log_pX = - self.val_X_logp(-logps.X)
        return log_pX

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # Compute gamma_s and gamma_t via the network. gamma is 噪声系数
        gamma_s = diffusion_utils.inflate_batch_array(self.gamma(s_float), X.size())  # (bs, 1, 1)
        gamma_t = diffusion_utils.inflate_batch_array(self.gamma(t_float), X.size())
        # (bs, 1)

        # Compute alpha_t and sigma_t from gamma, with correct size for X, E and z， alpha是
        alpha_t = diffusion_utils.alpha(gamma_t, X.size())  # (bs, 1, ..., 1), same n_dims than X
        sigma_t = diffusion_utils.sigma(gamma_t, X.size())  # (bs, 1, ..., 1), same n_dims than X

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = diffusion_utils.sample_feature_noise(X.size(), E.size(), y.size(), node_mask).type_as(X)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        X_t = alpha_t * X + sigma_t * eps.X


        noisy_data = {'t': t_int, 's': s_float, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
                      'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
                      'X_t': X_t, 'E_t': E, 'y_t': y, 'node_mask': node_mask}

        return noisy_data


    def compute_val_loss(self, pred, noisy_data, X, E, y, extra_x,node_mask, pos, edge_attr, test=False):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']
        s = noisy_data['s']
        gamma_s = noisy_data['gamma_s']  # gamma_s.size() == X.size()
        gamma_t = noisy_data['gamma_t']
        epsX = noisy_data['epsX']
        epsE = noisy_data['epsE']
        epsy = noisy_data['epsy']
        X_t = noisy_data['X_t']
        E_t = noisy_data['E_t']
        y_t = noisy_data['y_t']

        # 1.
        # 3. Diffusion loss

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization.
        SNR_weight = - (1 - diffusion_utils.SNR(gamma_s - gamma_t))
        sqrt_SNR_weight = torch.sqrt(SNR_weight)            # same n_dims than X
        # Compute the error.
        weighted_predX_diffusion = sqrt_SNR_weight * pred.X
        weighted_epsX_diffusion = sqrt_SNR_weight * epsX

        # Compute the MSE summed over channels
        if test:
            diffusion_error = self.test_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion)
        else:
            diffusion_error = self.val_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion)

        loss_all_t = 0.5 * self.T * diffusion_error           # t=0 is not included here.

        # 4. Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(s)  # bs, 1
        gamma_0 = diffusion_utils.inflate_batch_array(self.gamma(t_zeros), X_t.size())  # bs, 1, 1
        alpha_0 = diffusion_utils.alpha(gamma_0, X_t.size())  # bs, 1, 1
        sigma_0 = diffusion_utils.sigma(gamma_0, X_t.size())  # bs, 1, 1

        # Sample z_0 given X, E, y for timestep t, from q(z_t | X, E, y)
        eps0 = diffusion_utils.sample_feature_noise(X_t.size(), E_t.size(), y_t.size(), node_mask).type_as(X_t)

        X_0 = alpha_0 * X_t + sigma_0 * eps0.X
        E_0 = alpha_0.unsqueeze(1) * E_t + sigma_0.unsqueeze(1) * eps0.E
        y_0 = alpha_0.squeeze(1) * y_t + sigma_0.squeeze(1) * eps0.y

        noisy_data0 = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': t_zeros}
        extra_data = self.compute_extra_data(noisy_data, extra_x)
        pred_0 = self.forward(pos, noisy_data, extra_data, node_mask, edge_attr)

        loss_term_0 = - self.reconstruction_logp(data={'X': X, 'E': E, 'y': y},
                                                 data_0={'X_0': X_0, 'E_0': E_0, 'y_0': y_0},
                                                 gamma_0=gamma_0,
                                                 eps={'eps_X0': eps0.X, 'eps_E0': eps0.E, 'eps_y0': eps0.y},
                                                 pred_0=pred_0,
                                                 node_mask=node_mask,
                                                 test=test)

        # Combine terms
        nlls = loss_all_t + loss_term_0
        # assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll

        nll = self.test_nll(nlls) if test else self.val_nll(nlls)  # Average over the batch
        # Average over the batch

        wandb.log({"Estimator loss terms": loss_all_t.mean(),
                #    "log_pn": log_pN.mean(),
                   "loss_term_0": loss_term_0,
                   'test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, pos, noisy_data, extra_data, node_mask, edge_attr):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        if self.cfg.model.model == 'graph_tf':
            return self.model(X, E, y, node_mask)
        elif self.cfg.model.model == 'EGNN':
            return self.model(X, pos, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int==100)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y



        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            if i < 3:
                print("Example of generated E: ", atom_types)
                print("Example of generated X: ", edge_types)

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            predicted_graph_list.append([atom_types, edge_types])


        # Visualize chains
        if self.visualization_tools is not None:
            print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.visualization_tools.visualize(result_path, predicted_graph_list, save_final, log='predicted')
            print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, t, X_t, E_t, y_t, node_mask, last_step: bool):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        if last_step:
            predicted_graph = diffusion_utils.sample_discrete_features(pred_X, pred_E, node_mask=node_mask)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), \
               predicted_graph if last_step else None

    def compute_extra_data(self, noisy_data, extra_x):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)
        try:
            extra_X = torch.cat((extra_features.X, extra_molecular_features.X, extra_x), dim=-1)
        except AttributeError:
            extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)

        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
def generate_noise(self, shape):
    """ Generate noise for the given shape. """
    return diffusion_utils.sample_feature_noise(shape).type_as(self.model.parameters().next())

def forward_with_noise(self, pos, noise, extra_data, node_mask, edge_attr):
    """ Forward pass with noise. """
    X = torch.cat((noise['X_t'], extra_data.X), dim=2).float()
    E = torch.cat((noise['E_t'], extra_data.E), dim=3).float()
    y = torch.hstack((noise['y_t'], extra_data.y)).float()
    if self.cfg.model.model == 'graph_tf':
        return self.model(X, E, y, node_mask)
    elif self.cfg.model.model == 'EGNN':
        return self.model(X, pos, E, y, node_mask)

def noise_loss(self, noise, pred, true_X, true_E, true_y):
    """ Compute the loss using noise. """
    mse_loss = nn.MSELoss()
    loss_X = mse_loss(pred.X, true_X + noise.X)
    loss_E = mse_loss(pred.E, true_E + noise.E)
    loss_y = mse_loss(pred.y, true_y + noise.y)
    return loss_X + loss_E + loss_y