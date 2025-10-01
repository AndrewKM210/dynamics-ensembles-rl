import os
import random
import numpy as np
import torch
import gym
import d4rl  # noqa: F401

def seed_torch(seed=123):
    """ Sets all necessary seeds and makes torch deterministic """
    # environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)  # set python seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # set python seed
    np.random.seed(seed)  # set numpy seed
    torch.manual_seed(seed)  # sets torch seed for all devices (CPU and CUDA)
    torch.backends.cudnn.deterministic = True  # make cudnn deterministic
    torch.backends.cudnn.benchmark = False  # make cudnn deterministic


# Function adapted from Rajeswaran et al. (2017): https://github.com/aravindr93/mjrl/tree/v2
def d4rl2paths(dataset):
    """
    Convert d4rl dataset to paths (list of dictionaries)
    :param dataset: name of D4RL dataset
    :return: list of trajectories where each trajectory is a dictionary
    """
    e = gym.make(dataset)
    d = e.get_dataset()
    assert 'timeouts' in d.keys()
    num_samples = d['observations'].shape[0]
    timeouts = [t+1 for t in range(num_samples) if (d['timeouts'][t] or d['terminals'][t])]
    if timeouts[-1] != d['observations'].shape[0]:  
        timeouts.append(d['observations'].shape[0])
    timeouts.insert(0, 0)
    paths = []
    for idx in range(len(timeouts) - 1):
        path = dict()
        for key in d.keys():
            if 'metadata' not in key: 
                path[key] = d[key][timeouts[idx]: timeouts[idx+1]]
        paths.append(path)
    return paths