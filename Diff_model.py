# Modified SEDR class with Masked Reconstruction Loss and Diffusion Consistency Loss

import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .Diff_module import SEDR_module, SEDR_impute_module
from tqdm import tqdm


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def masked_reconstruction_loss(decoded, original, mask):
    loss_func = torch.nn.MSELoss(reduction='none')
    loss = loss_func(decoded, original)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def diffusion_consistency_loss(latent, diffusion_affinity):
    sim = torch.mm(latent, latent.t())
    sim = F.normalize(sim, p=1, dim=1)
    return F.mse_loss(sim, diffusion_affinity)


def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


class DiffScope:
    def __init__(
            self,
            X,
            graph_dict,
            rec_w=10,
            gcn_w=0.1,
            self_w=1,
            dec_kl_w=1,
            dcl_w=0.1,
            mask_recon=True,
            mode='clustering',
            device='cuda:0',
    ):
        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.self_w = self_w
        self.dec_kl_w = dec_kl_w
        self.dcl_w = dcl_w
        self.mask_recon = mask_recon
        self.device = device
        self.mode = mode

        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]
        self.cell_num = len(X)

        self.adj_norm = graph_dict['adj_norm'].to(self.device)
        self.adj_label = graph_dict['adj_label'].to(self.device)
        self.norm_value = graph_dict['norm_value']
        self.diffusion_affinity = graph_dict.get('diffusion_affinity', None)

        if 'mask' in graph_dict:
            self.mask = True
            self.adj_mask = graph_dict['mask'].to(self.device)
        else:
            self.mask = False

        if self.mask_recon:
            self.input_mask = (torch.rand_like(self.X) > 0.2).float()
            self.X_masked = self.X * self.input_mask
        else:
            self.X_masked = self.X

        if self.mode == 'clustering':
            self.model = SEDR_module(self.input_dim).to(self.device)
        elif self.mode == 'imputation':
            self.model = SEDR_impute_module(self.input_dim).to(self.device)
        else:
            raise ValueError(f'{self.mode} is not currently supported!')

    def mask_generator(self, N=1):
        idx = self.adj_label.indices()
        list_non_neighbor = []
        for i in range(self.cell_num):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N
            total_idx = torch.arange(0, self.cell_num, dtype=torch.float32).to(self.device)
            non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), dtype=torch.float32).to(self.device)
            random_non_neighbor = non_neighbor[indices[:n_selected].long()]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(self.adj_label.indices()[0], N)
        y = torch.cat(list_non_neighbor)

        indices = torch.stack([x, y])
        indices = torch.cat([self.adj_label.indices(), indices], dim=1)

        value = torch.cat([self.adj_label.values(), torch.zeros(len(x), dtype=torch.float32).to(self.device)])
        adj_mask = torch.sparse_coo_tensor(indices, value)

        return adj_mask

    def train_without_dec(self, epochs=200, lr=0.01, decay=0.01, N=1):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        self.model.train()

        for _ in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, _, _, _, loss_self = self.model(self.X_masked, self.adj_norm)

            if not self.mask:
                self.adj_mask = self.mask_generator(N=1)
                self.mask = True

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )

            if self.mask_recon:
                loss_rec = masked_reconstruction_loss(de_feat, self.X, self.input_mask)
            else:
                loss_rec = F.mse_loss(de_feat, self.X)

            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn + self.self_w * loss_self

            if self.diffusion_affinity is not None:
                loss_dcl = diffusion_consistency_loss(latent_z, self.diffusion_affinity)
                loss += self.dcl_w * loss_dcl

            loss.backward()
            self.optimizer.step()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print(f'Saving model to {save_model_file}')

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print(f'Loading model from {save_model_file}')

    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        return latent_z.cpu().numpy(), q.cpu().numpy(), feat_x.cpu().numpy(), gnn_z.cpu().numpy()

    def recon(self):
        self.model.eval()
        _, _, _, de_feat, _, _, _, _ = self.model(self.X, self.adj_norm)
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(de_feat.cpu().numpy())

    def train_with_dec(self, epochs=200, dec_interval=20, dec_tol=0.00, N=1):
        self.train_without_dec()

        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, n_init=self.model.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()

        for epoch_id in tqdm(range(epochs)):
            if epoch_id % dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if epoch_id > 0 and delta_label < dec_tol:
                    print(f'delta_label {delta_label:.4f} < tol {dec_tol}, stopping training.')
                    break

            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, out_q, _, _, _ = self.model(self.X_masked, self.adj_norm)

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )

            if self.mask_recon:
                loss_rec = masked_reconstruction_loss(de_feat, self.X, self.input_mask)
            else:
                loss_rec = F.mse_loss(de_feat, self.X)

            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device), reduction='batchmean')
            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn + self.dec_kl_w * loss_kl

            if self.diffusion_affinity is not None:
                loss_dcl = diffusion_consistency_loss(latent_z, self.diffusion_affinity)
                loss += self.dcl_w * loss_dcl

            loss.backward()
            self.optimizer.step()
