import argparse
import ast
import os
import time
import datetime
import pickle
import mlflow
import numpy as np
import yaml
from tabulate import tabulate
import utils
from models.deterministic_model import DeterministicModel
from models.probabilistic_model import ProbabilisticModel


parser = argparse.ArgumentParser(description="Train ensemble of neural networks")
parser.add_argument("--dataset", type=str, required=True, help="name of D4RL dataset")
parser.add_argument("--dataset_path", type=str, help="path to D4RL dataset (will save/load dataset)")
parser.add_argument("--config", type=str, required=True, help="path to config yaml file")
parser.add_argument("--output", type=str, default="test.pickle", help="output path for model")
parser.add_argument("--holdout_ratio", type=float, default=0.0, help="percentage of dataset to use for validation")
parser.add_argument("--csv", type=str, default=None, help="path to csv with training metrics, will track all metrics")
parser.add_argument("--track_training", action="store_true", default=False, help="track all training metrics")
parser.add_argument("--cpu", action="store_true", default=False, help="use CPU instead of GPU")
parser.add_argument(
    "--params", type=str, nargs="+", default=None, help="replaces config file parameters, format: param=value"
)
args = parser.parse_args()

# Open and parse config file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
config["fit_lr"] = float(config["fit_lr"])
config["hidden_size"] = ast.literal_eval(config["hidden_size"])

# Parse additional arguments
if args.params is not None:
    config = utils.parse_params(args.params, config)

# If asked, prepare folder where the csv should be saved
if args.csv is not None:
    utils.prepare_log_csv(args.csv)

# Seed libraries
seed = config.pop("seed")
utils.seed_torch(seed)

# Load dataset into a collection of paths
print("Loading dataset", args.dataset)
if args.dataset_path and os.path.exists(args.dataset_path):
    print("Loading from", args.dataset_path)
    paths = pickle.load(open(args.dataset_path, "rb"))
else:
    paths = utils.d4rl2paths(args.dataset)
    if args.dataset_path:
        pickle.dump(paths, open(args.dataset_path, "wb"))

# Select device
device = "cpu" if args.cpu else "cuda"

# Convert paths into training data (s, a, r, sp)
s = np.concatenate([p["observations"][:-1] for p in paths])
a = np.concatenate([p["actions"][:-1] for p in paths])
r = np.concatenate([p["rewards"][:-1] for p in paths])
sp = np.concatenate([p["observations"][1:] for p in paths])
s_h, a_h, sp_h = None, None, None
if args.holdout_ratio > 0:
    assert args.holdout_ratio >= 0 and args.holdout_ratio < 1.0, "Holdout ratio must be between 0 and 1"
    idx_rand = np.random.permutation(len(paths)).astype(int)
    num_paths_train = int(len(paths) * (1 - args.holdout_ratio))
    paths_train = [paths[i] for i in idx_rand[:num_paths_train]]
    paths_holdout = [paths[i] for i in idx_rand[num_paths_train:]]
    s = np.concatenate([p["observations"][:-1] for p in paths_train])
    a = np.concatenate([p["actions"][:-1] for p in paths_train])
    r = np.concatenate([p["rewards"][:-1] for p in paths_train])
    sp = np.concatenate([p["observations"][1:] for p in paths_train])
    s_h = np.concatenate([p["observations"][:-1] for p in paths_holdout])
    a_h = np.concatenate([p["actions"][:-1] for p in paths_holdout])
    sp_h = np.concatenate([p["observations"][1:] for p in paths_holdout])

# Show dataset statistics
avg_return = np.mean([np.sum(p["rewards"]) for p in paths])
print("\n", tabulate([("samples", s.shape[0]), ("avg_return", int(avg_return))], headers=["Dataset info", ""]))

# Show ensemble parameters
env = args.dataset.split("-")[0].lower()
dataset = "-".join(args.dataset.split("-")[1:]).lower()
config = {**{"environment": env, "dataset": dataset}, **config}
print("\n", tabulate([(k, v) for k, v in config.items()], headers=["Ensemble parameters", ""]))

# Initialize ensemble
print("\nFitting the ensemble to the dataset")
ensemble = []
model_class = DeterministicModel if not config["probabilistic"] else ProbabilisticModel
ensemble = [
    model_class(s.shape[-1], a.shape[-1], device=device, seed=seed + i, id=i, **config)
    for i in range(config["ensemble_size"])
]

mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns")
mlflow.set_experiment("dynamics_ensembles")
t_start = time.time()
track_metrics = args.track_training or args.csv is not None
fit_epochs = config["fit_epochs"]
with mlflow.start_run(run_name=args.dataset.lower()) as run:
    mlflow.log_params(config)
    for i, nn in enumerate(ensemble):
        print(f"\nFitting neural network {i}\n")
        metrics = nn.fit_dynamics(s, a, sp, s_h, a_h, sp_h, track_metrics=track_metrics, fit_epochs=fit_epochs)
        if args.csv is not None:
            utils.update_metrics_csv(metrics, args.csv, replace=i == 0)

# Save ensemble
print("\nSaving ensemble to", args.output)
pickle.dump(ensemble, open(args.output, "wb"))

# Show execution time
print("\nDone in", datetime.timedelta(seconds=round(time.time() - t_start)))
