# Learning Dynamics with Neural Network Ensembles

This repository contains the implementation of my MSc thesis work on learning environment dynamics with neural network ensembles.
The project explores deterministic vs probabilistic ensemble models for dynamics learning on D4RL benchmark datasets, and tracks training/evaluation with MLflow.

# Project Overview

- Implements ensemble neural networks for dynamics modeling:
  - Deterministic ensembles
  - Probabilistic ensembles
- Evaluates performance on D4RL datasets (commonly used for offline RL 
research).
- Logs metrics and artifacts using MLflow for experiment tracking.

The goal is to analyze how ensemble type (deterministic vs probabilistic) affects model quality and stability in learned dynamics models.

Repository Structure
```bash
.
├─ learn_model.py # Learns an ensemble of dynamics models
├─ dynamics_model.py # Definition of general dynamics model
├─ deterministic_model.py # Definition of deterministic model
├─ probabilistic_model.py # Definition of probabilistic model
├─ utils.py # Generic functions and classes
├─ configs/ # Configuration files
├─ mlruns/ # Will contain tracked MLflow runs
├─ logs/ # Saved csv logs
├─ requirements.txt # Python dependancies
└── README.md
```

# Installation

The project was tested with Python 3.10.18. It is recommended to use ```pyenv``` to create a new virtualenv.

```bash
git clone https://github.com/AndrewKM210/dynamics-ensembles-rl.git
cd dynamics-ensembles-rl
pip install -r requirements.txt

```

# Usage

The available datasets can be found in the Gym section of the following link:
- https://github.com/Farama-Foundation/d4rl/wiki/Tasks

Learn the dynamics of a D4RL dataset with a configuration file and store to file:

```bash
python learn_model.py --dataset hopper-medium-v0 --config configs/hopper_dnn.yaml --output model.pkl
```

Use a holdout split of the dataset to track validation loss:
```bash
python learn_model.py --dataset hopper-medium-v0 --config configs/hopper_dnn.yaml --holdout_ratio 0.2 --track_training
```

With ```--track_training``` all training metrics wil be logged in each epochs. Otherwise, the script prioritizes execution time and skips expensive non-essential metrics. The metrics are logged to MLflow. To view them use the following command and open http://127.0.0.1:5000: 

```bash
mlflow ui --port 5000
```

# Example Results

TODO

# References

- Fu et al. D4RL: Datasets for Deep Data-Driven Reinforcement Learning. NeurIPS 2020.
- Chua et al. Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS). NeurIPS 2018.
- MLflow: https://mlflow.org