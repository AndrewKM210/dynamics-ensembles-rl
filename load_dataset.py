import os
import models.utils as utils
import pickle
import argparse


def create_d4rl_dataset(dataset, output):
    """Create dataset as a collection of paths"""
    print("Downloading and preparing D4RL dataset")
    paths = utils.d4rl2paths(dataset)
    print("Saving dataset to", output)
    parent_dir = os.path.split(output)[0]
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    pickle.dump((paths, {"dataset": dataset}), open(output, "wb"))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn dynamics ensemble")
    parser.add_argument("--dataset", type=str, required=True, help="name of D4RL dataset")
    parser.add_argument("--output", type=str, help="path to save dataset")
    args = parser.parse_args()

    create_d4rl_dataset(args.dataset, args.output)
