import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from copy import deepcopy
import time

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor):
        return x.view(x.size(0), *self.shape)

class NNAgent(nn.Module):
    def __init__(self, bet_sizes: list[float], lr: float, update_target_every: int):
        super(NNAgent, self).__init__()
        self.bet_sizes = bet_sizes
        self.q_net = None
        self.lr = lr
        self.update_target_every = update_target_every

    def initialize_parameters(self, infostates: int):
        if self.q_net is None:
            self.create_network(infostates)
            self.make_clone_of_network_parameters()


    def take_action_PBS(self, state: torch.Tensor, infostates: int) -> torch.Tensor:
        self.initialize_parameters(infostates)
        actions = self.policy_net(state).detach()
        # add noise as exploration
        actions = actions + torch.randn_like(actions) * 0.1
        actions = torch.softmax(5 * actions, dim=-1)
        return actions
    
    def output_policy(self, state: torch.Tensor, infostates: int) -> torch.Tensor:
        self.initialize_parameters(infostates)
        actions = self.policy_net(state)
        return actions
    
    def output_value(self, state: torch.Tensor, infostates: int) -> torch.Tensor:
        self.initialize_parameters(infostates)
        actions = self.q_net(state)
        return actions


    def create_network(self, infostates: int) -> nn.Module:
        action_space = (infostates, len(self.bet_sizes) + 2)
        neurons = 64

        self.policy_net = nn.Sequential(nn.LazyLinear(neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, action_space[0] * action_space[1]),
            View(*action_space), 
            nn.Softmax(dim=-1)
        )

        self.optimizer_policy_net = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.q_net = nn.Sequential(nn.LazyLinear(neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, infostates),
        )

        self.optimizer_q_net = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def make_clone_of_network_parameters(self):
        self.q_net_clone = deepcopy(self.q_net)
        self.policy_net_clone = deepcopy(self.policy_net)
    

    def train_DDPG(self, data: tuple[Tensor], trains_done: int):
        if (trains_done + 1) % self.update_target_every == 0:
            self.make_clone_of_network_parameters()
        previous_state, actions, state, reward, dones = data
        # player_to_act = previous_state[:, 0]
        q_value = self.output_value(torch.cat((previous_state, actions.flatten(1)), 1), len(actions[0]))
        # print(actions[0])
        # print(reward[0])
        # print(q_value[0])
        # time.sleep(1)
        new_action = self.policy_net_clone(state)
        q_value_next = self.q_net_clone(torch.cat((state, new_action.flatten(1)), 1))

        target = (reward + q_value_next * (~dones.unsqueeze(1))).detach()

        loss = F.mse_loss(q_value, target)
        loss.backward()
        self.optimizer_q_net.step()
        self.optimizer_q_net.zero_grad()


        old_action = self.output_policy(previous_state, len(actions[0]))
        q_values = self.q_net(torch.cat((previous_state, old_action.flatten(1)), 1))
        loss_policy = -torch.mean(q_values)
        loss_policy.backward()
        self.optimizer_policy_net.step()
        self.optimizer_policy_net.zero_grad()

