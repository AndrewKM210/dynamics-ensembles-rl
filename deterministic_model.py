import numpy as np
import torch
from dynamics_model import DynamicsNN, DynamicsModel, swish


class DeterministicNN(DynamicsNN):
    def __init__(self, s_dim, a_dim, *args, **kwargs):
        super(DeterministicNN, self).__init__(s_dim, a_dim, *args, **kwargs)

    def forward(self, s, a):
        assert s.dim() == a.dim(), f"s and a dimensions differ: {s.dim()}, {a.dim()}"
        assert s.shape[0] == a.shape[0], f"s and a samples differ: {s.shape[0]}, {a.shape[0]}"

        # Normalize inputs
        s_in = (s - self.s_shift) / (self.s_scale + 1e-8)
        a_in = (a - self.a_shift) / (self.a_scale + 1e-8)

        # Feed through newtork
        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = self.activation_fn(out)
        out = self.layers[-1](out)

        # Transform the output (when not training)
        if self.transform_out:
            out = out * (self.out_scale + 1e-8) + self.out_shift
            out = out * self.mask if self.use_mask else out
            out = out + s if self.residual else out

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

    def predict(self, s, a):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        s_next = self.nn.forward(s, a)
        s_next = s_next.to("cpu").data.numpy()
        return s_next

    def predict_batched(self, s, a, batch_size=256):
        # Batch predict to lessen GPU usage
        num_samples = s.shape[0]
        num_steps = int(num_samples // batch_size) + 1
        s_next = np.ndarray((s.shape))
        for mb in range(num_steps):
            batch_idx = slice(mb * batch_size, (mb + 1) * batch_size)
            s_batch = torch.from_numpy(s[batch_idx]).float()
            a_batch = torch.from_numpy(a[batch_idx]).float()
            s_batch = s_batch.to(self.device)
            a_batch = a_batch.to(self.device)
            s_next_batch = self.nn.forward(s_batch, a_batch)
            s_next_batch = s_next_batch.to("cpu").data.numpy()
            s_next[batch_idx] = s_next_batch
        return s_next
