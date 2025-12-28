# Reinforcement Learning with Bayesian Linear Regression

This repository experiments with principles of Bayesian Inference and how they can be used in a reinforcement learning scenario. The structure of the agent is the same as the one proposed in this paper [Efficient Exploration through Bayesian Deep Q-Networks](https://arxiv.org/abs/1802.04412)

This repository extends this paper further by testing different policies given the information supplied by the Bayesian inference performed utilising the statistics in different ways.

Packages required:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy
pip install gymnasium ale-py opencv-python
```

Also need to create folders called "videos" and "results".

This repository has four strategies:

1. Thompson sampling over the weight distribution
2. Thompson sampling over the predictive distribution
3. Deterministic upper confidence bound
4. Stochastic upper confidence bound

The first strategy samples a weight from the current posterior weight distribution, calculates q-values for each action according to this sampled weight and the current state and chooses the action with the highest q-value.

The second strategy calculates the predictive distribution for q-values for each action, samples a q-value for each distribution and picks the action with highest sampled q-value.

The third strategy calculates the predictive distribution in the same way as strategy 2 but then performs te calculation $\mu + \kappa \sigma^2$ where $\mu, \sigma$ are the parameters of the posterior distribution and $\kappa$ is a parameter that gives weight to uncertainty. This $\kappa$ is reduced overtime. The action is then chosen with the hishest upper confidence bound.

The fourth strategy is embedded within the third strategy and can be used by setting `self.stochastic = True`. It performs the same sum as strategy 3 but then softmaxes these values across the actions. An action is then sampled according to this categorical distribution.

This model can be trained on either the tetris environment or the cartpole environment. In 'main.py' write `opt = Parameters(True)` for tetris and `opt = Parameters(False)` for cartpole. Parameters for either environment can also be set in the 'parameters.py' module.

To experiment with this repository you can use this line:

```
git clone https://github.com/alfielenton/BLRagent.git
```