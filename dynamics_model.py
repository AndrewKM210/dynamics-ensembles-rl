import torch
import numpy as np


def swish(x):
    """swish activation function"""
    return x * torch.sigmoid(x)


class DynamicsNN(torch.nn.Module):
    def __init__(self, s_dim, a_dim, out_dim, hidden_size, activation_fn, seed):
        super(DynamicsNN, self).__init__()

        # Neural network layers dimensions
        self.s_dim, self.a_dim, self.hidden_size = s_dim, a_dim, hidden_size
        self.out_dim = out_dim
        self.layer_sizes = (self.s_dim + self.a_dim,) + hidden_size + (self.out_dim,)

        # Neural network layers
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        )

        # Activation function
        self.activation_fn = activation_fn

        # Output scaling
        self.transform_out = True

    def set_transformations(self, s, a, sp, device):
        """Sets input and output scaling with the Mean Absolute Difference"""
        self.s_shift = np.mean(s, axis=0)
        self.a_shift = np.mean(a, axis=0)
        self.s_scale = np.mean(np.abs(s - self.s_shift), axis=0)
        self.a_scale = np.mean(np.abs(a - self.a_shift), axis=0)
        self.out_shift = np.mean(sp - s, axis=0)
        self.out_scale = np.mean(np.abs(sp - s - self.out_shift), axis=0)
        self.mask = self.out_scale >= 1e-8

    def transformations_to(self, device):
        self.s_shift = torch.from_numpy(self.s_shift).float().to(device)
        self.a_shift = torch.from_numpy(self.a_shift).float().to(device)
        self.s_scale = torch.from_numpy(self.s_scale).float().to(device)
        self.a_scale = torch.from_numpy(self.a_scale).float().to(device)
        self.out_shift = torch.from_numpy(self.out_shift).float().to(device)
        self.out_scale = torch.from_numpy(self.out_scale).float().to(device)
        self.mask = torch.from_numpy(self.mask).to(device)


class DynamicsModel:
    def __init__(self, device="cuda", *args, **kwargs):
        self.device = device
        self.mse_loss = torch.nn.MSELoss()
        self.holdout_idx = None

    def set_holdout_idx(self, holdout_idx):
        self.holdout_idx = holdout_idx

    def to(self, device):
        self.nn.to(device)

    def fit_dynamics(
        self, s, a, sp, s_h, a_h, sp_h, batch_size, fit_epochs, max_steps=1e4, track_mse=False, *args, **kwargs
    ):
        raise NotImplementedError("The method fit_dynamics must be implemented")

    def forward(self, s, a):
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) is np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        return self.nn.forward(s, a)

    def predict(self, s, a, to_cpu=True, det=True):
        raise NotImplementedError

    def predict_batched(self, s, a, batch_size=256, to_cpu=True, det=True):
        raise NotImplementedError

    def compute_loss_batched(self, s, a, sp, batch_size=256):
        assert type(s) is type(a) is type(sp)
        assert s.shape[0] == a.shape[0] == sp.shape[0]
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            sp = torch.from_numpy(sp).float()
        s = s.to(self.device)
        a = a.to(self.device)
        sp = sp.to(self.device)
        num_samples = s.shape[0]
        num_steps = int(num_samples // batch_size)
        losses = []
        for mb in range(num_steps):
            batch_idx = slice(mb * batch_size, (mb + 1) * batch_size)
            s_batch = s[batch_idx]
            a_batch = a[batch_idx]
            sp_batch = sp[batch_idx]
            mean = self.forward(s_batch, a_batch)
            if type(mean) is tuple:
                mean = mean[0]
            loss_batch = self.mse_loss(sp_batch, mean).to("cpu").data.numpy()
            losses.append(loss_batch)

        return np.mean(losses)
