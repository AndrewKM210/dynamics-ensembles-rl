# Configuration Files

Each configuration file contains the following parameters that can be customized:
- **hidden_size**: tuple with the size of each hidden layer
- **fit_epochs**: number of epochs to train each neural networks
- **fit_lr**: neural network Adam learning rate
- **scheduler**: learning rate scheduler, can be None or ExponentialLR
- **scheduler_gamma**: gamma parameter of the ExponentialLR scheduler
- **activation_fn**: neural network activation function, can be relu or swish
- **ensemble_size**: number of neural networks in ensemble
- **probabilistic**: If False use deterministic neural networks, if True use probabilistic ones
- **seed**: main seed for reproducibility 

The configuration files provided are tuned for the HalfCheetah, Hopper, Walker2d and Ant Gym MuJoCo environments:
- **halfcheetah_dnn.yaml**: deterministic neural network configuration for the HalfCheetah environment
- **halfcheetah_pnn.yaml**: probabilistic neural network configuration for the HalfCheetah environment
- **dnn.yaml**: deterministic neural network configuration for the rest of environments
- **pnn.yaml**: probabilistic neural network configuration for the rest of environments