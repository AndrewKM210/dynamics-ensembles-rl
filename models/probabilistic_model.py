import torch
from torch.nn.functional import softplus
import numpy as np
from models.dynamics_model import DynamicsNN, DynamicsModel, swish


class ProbabilisticNN(DynamicsNN):
    def __init__(self, s_dim, a_dim, *args, **kwargs):
        super(ProbabilisticNN, self).__init__(s_dim, a_dim, *args, **kwargs)

        # Probabilistic parameters
        self.inputs_mu = torch.nn.Parameter(torch.zeros(self.s_dim + self.a_dim), requires_grad=False)
        self.inputs_sigma = torch.nn.Parameter(torch.zeros(self.s_dim + self.a_dim), requires_grad=False)
        self.max_logvar = torch.nn.Parameter(torch.ones(1, self.out_dim // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = torch.nn.Parameter(-torch.ones(1, self.out_dim // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):
        decay_0 = 0.000025 * (self.layers[0].weight ** 2).sum()
        decay_1 = 0.00005 * (self.layers[1].weight ** 2).sum()
        decay_2 = 0.000075 * (self.layers[2].weight ** 2).sum()
        decay_3 = 0.000075 * (self.layers[3].weight ** 2).sum()
        decay_4 = 0.0001 * (self.layers[4].weight ** 2).sum()
        factor = 1
        decays = (decay_0 + decay_1 + decay_2 + decay_3 + decay_4) * factor
        return decays

    def _finalize_forward(self, s, out, ret_logvar=False):
        # Obtain mean and logvar
        mean = out[..., : self.out_dim // 2]
        logvar = out[..., self.out_dim // 2 :]
        logvar = self.max_logvar - softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + softplus(logvar - self.min_logvar)

        # Transform the output (when not training)
        if self.transform_out:
            mean *= self.out_scale + 1e-8
            mean += self.out_shift
            mean *= self.mask
            mean += s
            logvar *= self.out_scale + 1e-8
            logvar += self.out_shift
            logvar *= self.mask

        if ret_logvar:
            return mean, logvar
        return mean, torch.exp(logvar)


class ProbabilisticModel(DynamicsModel):
    def __init__(
        self,
        s_dim,
        a_dim,
        hidden_size,
        activation_fn="swish",
        fit_lr=5e-4,
        scheduler="ExponentialLR",
        scheduler_gamma=0.995,
        seed=123,
        device="cuda",
        *args,
        **kwargs,
    ):
        super(ProbabilisticModel, self).__init__(device=device, *args, **kwargs)
        assert len(hidden_size) == 4, "Probabilistic NNs must have 4 hidden layers"
        self.activation_fn = swish if activation_fn == "swish" else torch.relu
        out_dim = s_dim * 2  # Mean and logvar
        self.nn = ProbabilisticNN(s_dim, a_dim, out_dim, hidden_size, self.activation_fn, seed).to(device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=fit_lr)
        self.max_grad_norm = 200
        if scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)
        else:
            self.scheduler = None

    def _train_step(self, s, a, sp, n_batches, batch_size, n_samples, fill_last_batch):
        rand_idx = np.random.permutation(n_samples)
        gauss_loss = 0

        for b in range(n_batches):
            # Get batch of data, fill with random samples if last batch
            data_idx = rand_idx[b * batch_size : (b + 1) * batch_size]
            if b == n_batches - 1 and fill_last_batch:
                fill_size = (batch_size - data_idx.shape[0],)
                fill_idx = rand_idx[: b * batch_size][np.random.randint(0, b * batch_size, fill_size)]
                data_idx = np.concatenate([data_idx, fill_idx])

            # Move batch to GPU
            s_batch = torch.from_numpy(s[data_idx]).float().to(self.device)
            a_batch = torch.from_numpy(a[data_idx]).float().to(self.device)
            sp_batch = torch.from_numpy(sp[data_idx]).float().to(self.device)

            # Predict with neural network
            sp_hat, logvar = self.nn.forward(s_batch, a_batch, ret_logvar=True)

            # Compute losses
            inv_var = torch.exp(-logvar)
            train_losses = ((sp_hat - sp_batch) ** 2) * inv_var + logvar
            train_losses = train_losses.mean(-1).mean(-1).sum()
            decays = self.nn.compute_decays()
            logvar_reg = 0.01 * (self.nn.max_logvar.sum() - self.nn.min_logvar.sum())
            loss = train_losses + decays + logvar_reg

            # Optimizer step and gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Log metrics
            gauss_loss += train_losses.detach().cpu().numpy()
        gauss_loss = gauss_loss * 1.0 / n_batches
        return {"gauss_loss": gauss_loss}

    def predict(self, s, a, to_cpu=True, det=True):
        assert type(s) is type(a)
        assert s.shape[0] == a.shape[0]
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
        s, a = s.to(self.device), a.to(self.device)
        mean, logvar = self.nn.forward(s, a)
        pred = mean if det else mean + torch.randn_like(mean, device=self.device) * logvar.sqrt()
        pred = pred.detach().cpu().numpy() if to_cpu else pred
        return pred
