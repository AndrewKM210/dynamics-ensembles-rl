import torch
from torch.nn.functional import softplus
import numpy as np
from dynamics_model import DynamicsNN, DynamicsModel, swish
import mlflow


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

    def forward(self, s, a, ret_logvar=False):
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
        id=0,
        seed=123,
        device="cuda",
        *args,
        **kwargs,
    ):
        super(ProbabilisticModel, self).__init__(device=device, *args, **kwargs)
        assert len(hidden_size) == 4, "Probabilistic NNs must have 4 hidden layers"
        self.id = id
        self.activation_fn = swish if activation_fn == "swish" else torch.relu
        out_dim = s_dim * 2  # Mean and logvar
        self.nn = ProbabilisticNN(s_dim, a_dim, out_dim, hidden_size, self.activation_fn, seed).to(device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=fit_lr)
        self.max_grad_norm = 200
        if scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)
        else:
            self.scheduler = None

    def fit_dynamics(
        self,
        s,
        a,
        sp,
        s_h=None,
        a_h=None,
        sp_h=None,
        batch_size=256,
        fit_epochs=100,
        max_steps=1e4,
        track_mse=False,
        *args,
        **kwargs,
    ):
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
        mse_loss = 0
        val_loss = None

        # Start MLflow run
        mlflow.start_run(run_name=f"pnn-{self.id}", nested=True)

        # Train neural network
        for e in range(fit_epochs):
            print(f"Epoch: {e}")
            rand_idx = np.random.permutation(n_samples)
            gauss_loss = 0.0
            logvar_regularization = 0
            decay = 0.0
            logvar_means = []
            logvar_maxs = []
            inv_var_means = []
            inv_var_maxs = []

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
                logvar_regularization += logvar_reg.detach().cpu().numpy()
                decay += decays.detach().cpu().numpy()
                logvar_means.append(logvar.mean().cpu().data.numpy())
                logvar_maxs.append(logvar.max().cpu().data.numpy())
                inv_var_means.append(inv_var.mean().cpu().data.numpy())
                inv_var_maxs.append(inv_var.mean().cpu().data.numpy())

            # Update learning rate if using a scheduler
            lr = 0
            if self.scheduler:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

            # Compute validation loss if there is a holdout set
            if track_mse or e == 0 or e % 50 == 0 or e == fit_epochs - 1:
                mse_loss = self.compute_loss_batched(s, a, sp)
                if s_h is not None:
                    val_loss = self.compute_loss_batched(s_h, a_h, sp_h)

            # Log metrics to MLflow run
            gauss_loss = gauss_loss * 1.0 / n_batches
            logvar_regularization = logvar_regularization * 1.0 / n_batches
            decay = decay * 1.0 / n_batches
            print(f"mse_loss_{self.id}: {mse_loss}")
            mlflow.log_metric("mse_train_loss", mse_loss, step=e)
            print(f"gauss_loss_{self.id}: {gauss_loss}")
            mlflow.log_metric("gauss_loss", gauss_loss, step=e)
            print(f"logvar_mean_{self.id}: {np.mean(logvar_means)}")
            print(f"logvar_max_{self.id}: {np.max(logvar_maxs)}")
            print(f"inv_var_mean_{self.id}: {np.mean(inv_var_means)}")
            print(f"inv_var_max_{self.id}: {np.max(inv_var_maxs)}")
            print(f"logvar_regularization_{self.id}: {logvar_regularization}")
            print(f"decay_{self.id}: {decay}")
            if self.scheduler:
                print(f"lr_{self.id}: {lr}")
                mlflow.log_metric("lr", lr, step=e)
            if val_loss:
                print(f"val_loss_{self.id}: {val_loss}")
                mlflow.log_metric("mse_val_loss", val_loss, step=e)
            print()

        mlflow.end_run()
        self.nn.transform_out = True  # Once trained, outputs should not be normalized

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

    def predict_batched(self, s, a, batch_size=256, to_cpu=True, det=True):
        # Batch predict to lessen GPU usage
        assert type(s) is type(a)
        assert s.shape[0] == a.shape[0]
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
        num_samples = s.shape[0]
        num_steps = int(num_samples // batch_size) + 1
        pred = np.ndarray((s.shape))
        for mb in range(num_steps):
            batch_idx = slice(mb * batch_size, (mb + 1) * batch_size)
            s_batch = s[batch_idx].to(self.device)
            a_batch = a[batch_idx].to(self.device)
            mean, logvar = self.nn.forward(s_batch, a_batch)
            if det:
                pred_b = mean
            else:
                pred_b = mean + torch.randn_like(mean, device=self.device) * logvar.sqrt()
            pred_b = pred_b.to("cpu").data.numpy()
            pred[batch_idx] = pred_b
        return pred
