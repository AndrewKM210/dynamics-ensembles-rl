import argparse
import os
import pickle
import torch
import yaml
from models.deterministic_model import DeterministicModel
from models.probabilistic_model import ProbabilisticModel

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input", type=str, help="")
parser.add_argument("--output", type=str, help="")
args = parser.parse_args()

if os.path.isdir(args.input):
    out_path = args.output
    if os.path.exists(out_path):
        answer = input(f"File {out_path} already exists, continue? [Y\\n]: ").strip().lower()
        if answer != "y" and answer != "":
            print("Not converting ensemble")
            print("Done!")
            exit(0)
    parent_out_dir = os.path.split(out_path)[0]
    if not os.path.exists(parent_out_dir):
        print("Creating parent directory", parent_out_dir)
        os.mkdir(parent_out_dir)

    in_dir = args.input
    metadata_path = os.path.join(in_dir, "metadata.yaml")
    with open(metadata_path, "r") as f:
        print("Reading metadata from", metadata_path)
        metadata = yaml.safe_load(f)

    ensemble = []
    metadata["hidden_size"] = tuple(metadata["hidden_size"])
    for i in range(metadata["ensemble_size"]):
        print("\nLoading model", i)
        print("-------------")
        in_dir_i = os.path.join(in_dir, f"model_{i}")

        normalization_path = os.path.join(in_dir_i, "normalization.pt")
        print("Loading normalization parameters from", normalization_path)
        normalization = torch.load(normalization_path)
        s_dim = normalization["s_shift"].shape[0]
        a_dim = normalization["a_shift"].shape[0]

        pnn = metadata["probabilistic"]
        model_cls = ProbabilisticModel if pnn else DeterministicModel
        print("Creating model of class", model_cls)
        model = model_cls(s_dim=s_dim, a_dim=a_dim, **metadata)
        model.nn.s_shift = normalization["s_shift"]
        model.nn.a_shift = normalization["a_shift"]
        model.nn.s_scale = normalization["s_scale"]
        model.nn.a_scale = normalization["a_scale"]
        model.nn.out_shift = normalization["out_shift"]
        model.nn.out_scale = normalization["out_scale"]
        model.nn.mask = normalization["mask"]

        if pnn:
            extra_path = os.path.join(in_dir_i, "extra_params.pt")
            print("Loading extra parameters from", extra_path)
            extra = torch.load(extra_path)
            model.nn.inputs_mu.data.copy_(extra["inputs_mu"])
            model.nn.inputs_sigma.data.copy_(extra["inputs_sigma"])
            model.nn.max_logvar.data.copy_(extra["max_logvar"])
            model.nn.min_logvar.data.copy_(extra["min_logvar"])

        state_dict_path = os.path.join(in_dir_i, "state_dict.pt")
        print("Loading state_dict from", state_dict_path)
        model.nn.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

        device = "cpu" if metadata["cpu"] else "cuda"
        print("Moving reconstructed model to", "CPU" if device == "cpu" else "GPU")
        model.to(device)
        ensemble.append(model)

    parent_out_dir = os.path.split(out_path)[0]
    if not os.path.exists(parent_out_dir):
        print("Creating parent directory", parent_out_dir)
        os.mkdir(parent_out_dir)

    print("\nSaving reconstructed ensemble to ", out_path)
    pickle.dump((ensemble, metadata), open(out_path, "wb"))
    print("Done!")
else:
    (ensemble, metadata) = pickle.load(open(args.input, "rb"))

    out_dir = args.output
    if os.path.exists(out_dir):
        answer = input(f"Directory {out_dir} already exists, continue? [Y\\n]: ").strip().lower()
        if answer != "y" and answer != "":
            print("Not converting ensemble")
            print("Done!")
            exit(0)
    else:
        print("Creating directory", out_dir)
        os.mkdir(out_dir)

    metadata_path = os.path.join(out_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        print("Writing metadata to", metadata_path)
        print(metadata)
        yaml.safe_dump(metadata, f)

    for i, model in enumerate(ensemble):
        print(f"\nSaving model {i}")
        print("--------------")
        out_dir_i = os.path.join(out_dir, f"model_{i}")
        if not os.path.exists(out_dir_i):
            print("Creating directory", out_dir_i)
            os.mkdir(out_dir_i)
        state_dict_path = os.path.join(out_dir_i, "state_dict.pt")
        print("Saving state_dict to", state_dict_path)
        torch.save(model.nn.state_dict(), state_dict_path)

        pnn = metadata["probabilistic"]
        if pnn:
            extra = {
                "inputs_mu": model.nn.inputs_mu.data,
                "inputs_sigma": model.nn.inputs_sigma.data,
                "min_logvar": model.nn.min_logvar.data,
                "max_logvar": model.nn.max_logvar.data,
            }
            extra_path = os.path.join(out_dir_i, "extra_params.pt")
            print("Saving extra parameters to", extra_path)
            torch.save(extra, extra_path)

        model.nn.transformations_to("cpu")
        normalization = dict(
            s_shift=model.nn.s_shift,
            a_shift=model.nn.a_shift,
            s_scale=model.nn.s_scale,
            a_scale=model.nn.a_scale,
            out_shift=model.nn.out_shift,
            out_scale=model.nn.out_scale,
            mask=model.nn.mask,
        )
        normalization_path = os.path.join(out_dir_i, "normalization.pt")
        print("Saving normalization parameters to", normalization_path)
        torch.save(normalization, normalization_path)
    print("Done!")
