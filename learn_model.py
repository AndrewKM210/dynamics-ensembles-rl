
import numpy as np
from probabilistic_model import ProbabilisticModel
from deterministic_model import DeterministicModel
import pickle
import argparse
import utils

utils.seed_torch(123)

parser = argparse.ArgumentParser(description="Train ensemble of neural networks")
parser.add_argument("--dataset", type=str, required=True, help="name of D4RL dataset")
parser.add_argument("--output", type=str, default="test.pickle", help="output path for model")
parser.add_argument("--probabilistic", action='store_true', help="use probabilistic neural networks")
parser.add_argument("--holdout_ratio", type=float, default=0.0, help="percentage of dataset to use for validation")
args = parser.parse_args()

print('Loading dataset', args.dataset)
paths = utils.d4rl2paths(args.dataset)

# # TODO: remove
paths = paths[:100]

s = np.concatenate([p["observations"][:-1] for p in paths])
a = np.concatenate([p["actions"][:-1] for p in paths])
r = np.concatenate([p["rewards"][:-1] for p in paths])
sp = np.concatenate([p["observations"][1:] for p in paths])
s_h, a_h, sp_h = None, None, None
if args.holdout_ratio > 0:
    if args.holdout_ratio <= 0 or args.holdout_ratio >= 1.0:
        print("Error: holdout ratio must be between 0 and 1")
        exit(-1)
    idx_rand = np.random.permutation(len(paths)).astype(int)
    num_paths_train = int(len(paths)*(1-args.holdout_ratio))
    paths_train = [paths[i] for i in idx_rand[:num_paths_train]]
    paths_holdout = [paths[i] for i in idx_rand[num_paths_train:]]
    s = np.concatenate([p["observations"][:-1] for p in paths_train])
    a = np.concatenate([p["actions"][:-1] for p in paths_train])
    r = np.concatenate([p["rewards"][:-1] for p in paths_train])
    sp = np.concatenate([p["observations"][1:] for p in paths_train])
    s_h = np.concatenate([p["observations"][:-1] for p in paths_holdout])
    a_h = np.concatenate([p["actions"][:-1] for p in paths_holdout])
    sp_h = np.concatenate([p["observations"][1:] for p in paths_holdout])

avg_return = np.mean([np.sum(p["rewards"]) for p in paths])
print('Number of samples:', s.shape[0])
print('Average return:', avg_return)

print('\nFitting the ensemble to the dataset')
if not args.probabilistic:
    print('Using deterministic neural networks\n')
    dynamics_model = DeterministicModel(s.shape[-1], a.shape[-1], (512, 512), device='cpu')
    dynamics_model.fit_dynamics(s, a, sp, s_h, a_h, sp_h, track_val=True, epochs=300)
else:
    print('Using probabilistic neural networks\n')
    dynamics_model = ProbabilisticModel(s.shape[-1], a.shape[-1], (256, 256, 256, 256), device='cpu')
    dynamics_model.fit_dynamics(s, a, sp, s_h, a_h, sp_h, track_mse=True, epochs=300)

# pickle.dump(ensemble.models, open(args.output, "wb"))