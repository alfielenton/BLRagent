# Reinforcement Learning with Bayesian Linear Regression

This repository experiments with principles of Bayesian Inference and how they can be used in a reinforcement learning scenario. The structure of the agent is the same as the one proposed in this paper [Efficient Exploration through Bayesian Deep Q-Networks](https://arxiv.org/abs/1802.04412)

This repository extends this paper further by testing different policies given the information supplied by the Bayesian inference performed utilising the statistics in different ways.

Packages required:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy
pip install torch
pip install gymnasium ale-py opencv-python
```

Also need to create folders called "videos" and "results".

This repository has four strategies:

1. Thompson sampling over the weight distribution
2. Thompson sampling over the predictive distribution
3. Deterministic upper confidence bound
4. Stochastic upper confidence bound