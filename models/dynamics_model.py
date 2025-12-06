import torch
import numpy as np
import mlflow
from tabulate import tabulate
from utils import MetricsLog


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

    def _finalize_forward(self, s, out):
        raise NotImplementedError

    def forward(self, s, a, *args, **kwargs):
        assert s.dim() == a.dim(), f"s and a dimensions differ: {s.dim()}, {a.dim()}"
        assert s.shape[0] == a.shape[0], f"s and a samples differ: {s.shape[0]}, {a.shape[0]}"

        # Normalize inputs
        s_in = (s - self.s_shift) / (self.s_scale + 1e-8)
        a_in = (a - self.a_shift) / (self.a_scale + 1e-8)

        # Feed through network
        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = self.activation_fn(out)
        out = self.layers[-1](out)

        return self._finalize_forward(s, out, *args, **kwargs)


class DynamicsModel:
    def __init__(self, device="cuda", probabilistic=False, id=0, *args, **kwargs):
        self.probabilistic = probabilistic  # Neural network type
        self.device = device
        self.mse_loss = torch.nn.MSELoss()
        self.holdout_idx = None
        self.id = id

    def set_holdout_idx(self, holdout_idx):
        self.holdout_idx = holdout_idx

    def to(self, device):
        self.nn.to(device)

    def _train_step(self, s, a, sp, n_batches, batch_size, n_samples, fill_last_batch):
        raise NotImplementedError

    def fit_dynamics(self, s, a, sp, s_h, a_h, sp_h, fit_epochs, batch_size=256, max_steps=1e4, track_metrics=False):
        # Train on normalized data
        self.nn.transform_out = False
        self.nn.set_transformations(s, a, sp, self.device)
        sp = (sp - s - self.nn.out_shift) / (self.nn.out_scale + 1e-8)
        if sp_h is not None:
            sp_h = (sp_h - s_h - self.nn.out_shift) / (self.nn.out_scale + 1e-8)
        self.nn.transformations_to(self.device)

        # Compute number of batches
        n_samples = sp.shape[0]
        n_batches = int(n_samples // batch_size)
        fill_last_batch = False
        if n_samples != n_batches * batch_size:
            n_batches += 1
            fill_last_batch = True

        # Start MLflow run
        mlflow.start_run(run_name=f"{'pnn' if self.probabilistic else 'dnn'}-{self.id}", nested=True)
        metrics_log = MetricsLog()

        # Train neural network
        for e in range(fit_epochs):
            metrics = self._train_step(s, a, sp, n_batches, batch_size, n_samples, fill_last_batch)

            # Update learning rate if using a scheduler
            lr = 0
            if self.scheduler:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                metrics["lr"] = lr

            # Compute MSE if needed
            if track_metrics or e == 0 or e % 50 == 0 or e == fit_epochs - 1:
                metrics["train_mse"] = (
                    self.compute_loss_batched(s, a, sp) if self.probabilistic else metrics["mse_loss"]
                )
                if s_h is not None:
                    metrics["val_mse"] = self.compute_loss_batched(s_h, a_h, sp_h)

            # Log metrics to MLflow run
            mlflow.log_metrics(metrics, step=e)
            print(tabulate([(k, v) for k, v in metrics.items()], headers=[f"Epoch {e}", ""]))
            metrics_log.update({**{"step": e, "model_id": self.id}, **metrics})
            print()

        mlflow.end_run()
        self.nn.transform_out = True  # Once trained, outputs should not be normalized
        return metrics_log

    def forward(self, s, a):
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) is np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        return self.nn.forward(s, a)

    def predict(self, s, a, to_cpu=True, *arg, **kwarg):
        raise NotImplementedError

    def predict_batched(self, s, a, batch_size=256, to_cpu=True, *arg, **kwarg):
        # Batch predict to lessen GPU usage
        assert type(s) is type(a)
        assert s.shape[0] == a.shape[0]
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
        num_samples = s.shape[0]
        num_steps = int(num_samples // batch_size) + 1
        pred = np.ndarray((s.shape))
        for batch in range(num_steps):
            batch_idx = slice(batch * batch_size, (batch + 1) * batch_size)
            s_batch = s[batch_idx].to(self.device)
            a_batch = a[batch_idx].to(self.device)
            pred[batch_idx] = self.predict(s_batch, a_batch, to_cpu, *arg, **kwarg)
        return pred

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
            pred = self.predict(s_batch, a_batch, to_cpu=False)
            loss_batch = self.mse_loss(sp_batch, pred).to("cpu").data.numpy()
            losses.append(loss_batch)

        return np.mean(losses)
