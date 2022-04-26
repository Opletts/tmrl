from dataclasses import dataclass
import itertools
from copy import deepcopy

# third-party imports
import numpy as np
import torch
from torch.optim import Adam

# local imports
# import tmrl.custom.custom_models as core
import networks as core
from tmrl.nn import copy_shared, no_grad
from tmrl.util import cached_property
from tmrl.training import TrainingAgent
import logging

@dataclass(eq=0)
class DDPGAgent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.MyActorCriticModule
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    tau = 0.005
    loss = torch.nn.MSELoss()
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    # def __init__(self, observation_space, action_space, device):
    #     super().__init__(self, observation_space, action_space, device)

    #     observation_space, action_space = self.observation_space, self.action_space
    #     device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
    #     model = self.model_cls(observation_space, action_space)
    #     logging.debug(f" device DDPG: {device}")
    #     self.model = model.to(device)
    #     self.model_target = no_grad(deepcopy(self.model))

    #     # Set up optimizers for policy and q-function
    #     self.actor_optimizer = Adam(self.model.actor.parameters(), lr = self.lr_actor, weight_decay = 1e-4)
    #     self.critic_optimizer = Adam(self.model.q.parameters(), lr = self.lr_critic, weight_decay = 1e-4)

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device DDPG: {device}")
        
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
        self.actor_optimizer = Adam(self.model.actor.parameters(), lr = self.lr_actor, weight_decay = 1e-4)
        self.critic_optimizer = Adam(self.model.q.parameters(), lr = self.lr_critic, weight_decay = 1e-4)


    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d = batch

        pi = self.model_target.actor(o2)
        q_value_ = self.model_target.q([o2, pi])
        target = torch.unsqueeze(r, 1) + self.gamma*q_value_

        #Critic Update
        self.model.q.zero_grad()
        q_value = self.model.q([o, a])
        value_loss = self.loss(q_value, target)
        value_loss.backward()
        self.critic_optimizer.step()

        #Actor Update
        self.model.actor.zero_grad()
        new_policy_actions = self.model.actor(o)
        actor_loss = -self.model.q([o, new_policy_actions])
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            # If tau = 1 -> hard update (should only be done during init)
            tau = self.tau
            # Update Actor Network
            for target_param, param in zip(self.model_target.actor.parameters(), self.model.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            # Update Critic Network
            for target_param, param in zip(self.model_target.q.parameters(), self.model.q.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        ret_dict = dict(
            loss_actor=actor_loss.detach(),
            loss_critic=value_loss.detach(),
        )

        return ret_dict