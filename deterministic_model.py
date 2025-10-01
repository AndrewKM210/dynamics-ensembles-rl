import numpy as np
import torch
import torch.nn as nn
from dynamics_model import DynamicsNN, DynamicsModel, swish
import mlflow

class DeterministicNN(DynamicsNN):
    def __init__(self,s_dim, a_dim, *args, **kwargs):
        super(DeterministicNN, self).__init__(s_dim, a_dim, *args, **kwargs)

    def forward(self, s, a):
        if s.dim() != a.dim():
            print("Error: State and action inputs should be of the same size")
            exit(1)
        
        # normalize inputs
        s_in = (s - self.s_shift) / (self.s_scale + 1e-8)
        a_in = (a - self.a_shift) / (self.a_scale + 1e-8)

        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = self.activation_fn(out)
        out = self.layers[-1](out)
        
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
        activation_fn='relu',
        fit_lr=5e-4,
        scheduler='ExponentialLR',
        scheduler_gamma=0.99,
        id=0,
        seed=123,
        device='cuda',
        *args,
        **kwargs,
    ):
        super(DeterministicModel, self).__init__(device=device, *args, **kwargs)
        self.id = id
        self.activation_fn = swish if activation_fn == "swish" else torch.relu
        out_dim = s_dim
        self.nn = DeterministicNN(s_dim, a_dim, out_dim, hidden_size, self.activation_fn, seed=seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=fit_lr)

        if scheduler == "MultistepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[200, 250, 275], gamma=0.25
            )  # BRAC unscaled!
        elif scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)
        elif scheduler == "MultiplicativeLR":
            lmbda = lambda epoch: 0.99
            self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lmbda)
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
        epochs=100,
        max_steps=1e4,
        track_val=False,
    ):
        # logging metrics
        train_metrics = {
            f"mse_loss_{self.id}": [],
        }

        if s_h is not None:
            train_metrics.update({f"val_loss_{self.id}": []})

        self.nn.transform_out = False
        self.nn.set_transformations(s, a, sp, self.device)
        sp = (sp - s - self.nn.out_shift) / (self.nn.out_scale + 1e-8)
        if sp_h is not None:
            sp_h = (sp_h - s_h - self.nn.out_shift) / (self.nn.out_scale + 1e-8)
        self.nn.transformations_to(self.device)
        n_samples = sp.shape[0]
        n_batches = int(n_samples // batch_size)
        fill_last_batch = False
        if n_samples != n_batches * batch_size:
            n_batches += 1
            fill_last_batch = True
        val_loss = None

        # Experiment settings
        mlflow.set_experiment('dynamics_model')
        run = mlflow.start_run(run_name=f"hopper_ensemble_deterministic_{self.id}",
            tags={
                "dataset": "hopper-medium-v0",
                "model": "deterministic",
                "notes": "Deterministic neural network"
        })
        mlflow.log_params({
            "hidden_layers": 4,
            "hidden_size": 512,
            "lr": 5e-4,
            "epochs": 25
        })

        for e in range(epochs):
            print(f"Epoch: {e}")
            rand_idx = np.random.permutation(n_samples)
            mse_loss = 0.0

            for b in range(n_batches):
                # get batch of data, fill with random samples if last batch
                data_idx = rand_idx[b * batch_size : (b + 1) * batch_size]
                if b == n_batches - 1 and fill_last_batch:
                    fill_size = (batch_size - data_idx.shape[0],)
                    fill_idx = rand_idx[: b * batch_size][np.random.randint(0, b * batch_size, fill_size)]
                    data_idx = np.concatenate([data_idx, fill_idx])

                # move batch to GPU
                s_batch = torch.from_numpy(s[data_idx]).float().to(self.device)
                a_batch = torch.from_numpy(a[data_idx]).float().to(self.device)
                sp_batch = torch.from_numpy(sp[data_idx]).float().to(self.device)

                # predict
                sp_hat = self.nn.forward(s_batch, a_batch)
                
                loss = self.mse_loss(sp_hat, sp_batch)

                # optimizer step and clip gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log metrics
                mse_loss += loss.detach().cpu().numpy()

            # update learning rate if using a scheduler
            lr = 0
            if self.scheduler:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

            # compute validation loss if there is a holdout set
            if (e == 0 or e % 50 == 0 or e == epochs - 1 or track_val) and s_h is not None:
                val_loss = self.compute_loss_batched(s_h, a_h, sp_h)

            mse_loss = mse_loss * 1.0 / n_batches
            print(f"mse_loss_{self.id}: {mse_loss}")
            train_metrics[f"mse_loss_{self.id}"].append(mse_loss)
            mlflow.log_metric("mse_train_loss", mse_loss, step=e)
            if self.scheduler:
                print(f"lr_{self.id}: {lr}")
                mlflow.log_metric("lr", lr, step=e)
            if val_loss:
                print(f"val_loss_{self.id}: {val_loss}")
                train_metrics[f"val_loss_{self.id}"].append(val_loss)
                mlflow.log_metric("mse_val_loss", val_loss, step=e)
            print()

        mlflow.end_run()
        self.nn.transform_out = True
        return train_metrics

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
        num_steps = int(num_samples // batch_size)+1
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