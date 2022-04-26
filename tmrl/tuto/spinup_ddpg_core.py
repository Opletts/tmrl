import functools
import operator

import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
import torch
from torch.distributions.normal import Normal

class ActorModule(torch.nn.Module, ABC):
    """
    Interface for the RolloutWorker(s) to interact with the policy.

    This is a torch neural network and must implement forward().
    Typically, act() calls forward() with gradients turned off.

    The __init()__ definition must at least take the two following arguments (args or kwargs):
        observation_space
        action_space
    The __init()__ method must also call the superclass __init__() via super().__init__(observation_space, action_space)
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = None

    @abstractmethod
    def act(self, obs, test=False):
        """
        Returns an action from an observation.
        
        Args:
            obs: the observation
            test: bool: True at test time, False otherwise
        Returns:
            act: numpy array: the computed action
        """
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def to(self, device):
        """
        keeps track which device this module has been moved to
        """
        self.device = device
        return super().to(device=device)

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLPActor(ActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, act_buf_len=0):
        super().__init__(observation_space, action_space)
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        self.dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        pi_sizes = [dim_obs] + list(hidden_sizes) + [self.dim_act]
        self.pi = mlp(pi_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], self.dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], self.dim_act)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.

        # pi_action = self.act_limit * self.pi(torch.cat(obs, -1))
        net_out = self.pi(torch.cat(obs, -1))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        pi_action = pi_distribution.rsample()

        pi_action = pi_action.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action.squeeze()
    
    def act(self, obs, test=False):
        with torch.no_grad():
            a = self.forward(obs)
            print("ACTION BEFORE NOISE :", a)
            a += torch.Tensor(0.001 * np.random.randn(self.dim_act), dtype=torch.float32).to(self.device)
            a = torch.clamp(a, -1, 1)

            print("ACTION AFTER NOISE :", a)

            return a.numpy()


class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = MLPActor(observation_space, action_space, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs).numpy()
