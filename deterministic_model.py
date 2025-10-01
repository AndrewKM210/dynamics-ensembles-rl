import numpy as np
import torch
from dynamics_model import DynamicsNN, DynamicsModel, swish


class DeterministicNN(DynamicsNN):
    def __init__(self, s_dim, a_dim, *args, **kwargs):
        super(DeterministicNN, self).__init__(s_dim, a_dim, *args, **kwargs)

    def _finalize_forward(self, s, out):
        # Transform the output (when not training)
        if self.transform_out:
            out = out * (self.out_scale + 1e-8) + self.out_shift
            out = out * self.mask
            out = out + s
        return out


class DeterministicModel(DynamicsModel):
    def __init__(
        self,
        s_dim,
        a_dim,
        hidden_size,
        activation_fn="relu",
        fit_lr=5e-4,
        scheduler="ExponentialLR",
        scheduler_gamma=0.99,
        id=0,
        seed=123,
        device="cuda",
        *args,
        **kwargs,
    ):
        super(DeterministicModel, self).__init__(device=device, *args, **kwargs)
        self.id = id
        self.activation_fn = swish if activation_fn == "swish" else torch.relu
        out_dim = s_dim
        self.nn = DeterministicNN(s_dim, a_dim, out_dim, hidden_size, self.activation_fn, seed=seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=fit_lr)
        if scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)
        else:
            self.scheduler = None

    def _train_step(self, s, a, sp, n_batches, batch_size, n_samples, fill_last_batch):
        rand_idx = np.random.permutation(n_samples)
        mse_loss = 0.0

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
            sp_hat = self.nn.forward(s_batch, a_batch)

            # Compute loss (MSE)
            loss = self.mse_loss(sp_hat, sp_batch)

            # Optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mse_loss += loss.detach().cpu().numpy()

        mse_loss = mse_loss * 1.0 / n_batches
        return {"mse_loss": mse_loss}

    def predict(self, s, a, to_cpu=True):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        pred = self.nn.forward(s, a)
        pred = pred.to("cpu").data.numpy() if to_cpu else pred
        return pred
