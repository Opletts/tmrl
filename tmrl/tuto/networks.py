import torch
from tmrl.actor import ActorModule
from tmrl.util import prod
import torch.nn.functional as F
import numpy as np

def fanin_init(size, fanin = None):
        fanin = fanin or size[0]
        v = 1. / np.sqrt(fanin)
        return torch.Tensor(size).uniform_(-v, v)

class MyActorModule(ActorModule):
    """
    Directly adapted from the Spinup implementation of SAC
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        self.dim_act = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.act_noise_scale = 0.1

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'

        self.fc1_dims = 400
        self.fc2_dims = 300

        self.fc1 = torch.nn.Linear(dim_obs, self.fc1_dims)
        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = torch.nn.Linear(self.fc2_dims, self.dim_act)

        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.mu.weight.data.uniform_(-init_w, init_w)

    def forward(self, obs, test=False, with_logprob=True):

        obs = torch.cat(obs, -1)
        prob = F.relu(self.fc1(obs))
        prob = F.relu(self.fc2(prob))
        actions = self.mu(prob)
        actions = torch.tanh(actions)

        # Add noise for training mode
        if test == False:
            actions = actions + torch.tensor(self.act_noise_scale * np.random.randn(self.dim_act), dtype=torch.float32).to(self.device)

        actions = torch.tanh(actions)
        actions = self.act_limit * actions # we don't really need this, do we? tanh already caps it between -1 and 1
        return actions.squeeze()

    def act(self, obs, test=False):
        with torch.no_grad():

            a = self.forward(obs, test, False)

            return a.numpy()

class MyCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
        act_dim = action_space.shape[0]

        ip_size = [obs_dim + act_dim] + list((256, 256)) + [1]

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
    
        self.fc1_dims = 400
        self.fc2_dims = 300

        self.fc1 = torch.nn.Linear(ip_size[0], ip_size[1])
        self.fc2 = torch.nn.Linear(ip_size[1], ip_size[2])
        self.q = torch.nn.Linear(ip_size[2], 1)

        self.init_weights(3e-3)

        self.to(self.device)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.q.weight.data.uniform_(-init_w, init_w)

    def forward(self, X):
        state, action = X
        action_value = F.relu(self.fc1(torch.cat((*state, action), -1)))
        action_value = F.relu(self.fc2(action_value))
        action_value = self.q(action_value)
        return action_value

class MyActorCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # our ActorModule:
        self.actor = MyActorModule(observation_space, action_space)
        
        # Critic networks:
        self.q = MyCriticModule(observation_space, action_space)