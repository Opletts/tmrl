import itertools
import functools
import weakref
from copy import deepcopy
from dataclasses import InitVar, dataclass
from abc import ABC, abstractmethod

# third-party imports
import numpy as np
import torch
from torch.optim import Adam

# local imports
import spinup_ddpg_core as core
import logging

class cached_property:
    """Similar to `property` but after calling the getter/init function the result is cached.
    It can be used to create object attributes that aren't stored in the object's __dict__.
    This is useful if we want to exclude certain attributes from being pickled."""
    def __init__(self, init=None):
        self.cache = {}
        self.init = init

    def __get__(self, instance, owner):
        if id(instance) not in self.cache:
            if self.init is None: raise AttributeError()
            self.__set__(instance, self.init(instance))
        return self.cache[id(instance)][0]

    def __set__(self, instance, value):
        # Cache the attribute value based on the instance id. If instance is garbage collected its cached value is removed.
        self.cache[id(instance)] = (value, weakref.ref(instance, functools.partial(self.cache.pop, id(instance))))

class TrainingAgent(ABC):
    def __init__(self,
                 observation_space,
                 action_space,
                 device):
        """
        observation_space, action_space and device are here for your convenience.

        You are free to use them or not, but your subclass must have them as args or kwargs of __init__() .
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def train(self, batch):
        """
        Executes a training step.

        Args:
            batch: tuple or batched torch.tensors (previous observation, action, reward, new observation, done)

        Returns:
            ret_dict: dictionary: a dictionary containing one entry per metric you wish to log (e.g. for wandb)
        """
        raise NotImplementedError

    @abstractmethod
    def get_actor(self):
        """
        Returns the current ActorModule to be broadcast to the RolloutWorkers.

        Returns:
             actor: ActorModule: current actor to be broadcast
        """
        raise NotImplementedError

def no_grad(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def copy_shared(model_a):
    """Create a deepcopy of a model but with the underlying state_dict shared. E.g. useful in combination with `no_grad`."""
    model_b = deepcopy(model_a)
    sda = model_a.state_dict(keep_vars=True)
    sdb = model_b.state_dict(keep_vars=True)
    for key in sda:
        a, b = sda[key], sdb[key]
        b.data = a.data  # strangely this will not make a.data and b.data the same object but their underlying data_ptr will be the same
        assert b.storage().data_ptr() == a.storage().data_ptr()
    return model_b

@dataclass(eq=0)
class SpinupDDPGAgent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device DDPG: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.model.q.parameters(), lr=self.lr_critic)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d = batch

        q = self.model.q(o, a)

        with torch.no_grad():
            q_pi_targ = self.model_target.q(o2, self.model_target.actor(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup)**2).mean()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.model.q.parameters():
            p.requires_grad = False

        q_pi = self.model.q(o, self.model.actor(o))
        loss_pi = -q_pi.mean()
        
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        
        for p in self.model.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        ret_dict = dict(
            loss_actor=loss_pi.detach(),
            loss_critic=loss_q.detach(),
        )

        return ret_dict