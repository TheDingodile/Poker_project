import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor):
        return x.view(x.size(0), *self.shape)

class NNAgent(nn.Module):
    def __init__(self, bet_sizes: list[float], action_space_shape: list[int] = None):
        super(NNAgent, self).__init__()
        self.bet_sizes = bet_sizes
        self.back_bone = None


    def take_action_PBS(self, state: torch.Tensor, infostates: int) -> torch.Tensor:
        if self.network is None:
            self.backbone, self.value_head, self.policy_head = self.create_network(infostates)
        
        actions = self.network(state)
        return actions


    def create_network(self, infostates: int) -> nn.Module:
        action_space = (infostates, len(self.bet_sizes) + 2)
        neurons = 64
        back_bone = nn.Sequential(
            nn.LazyLinear(neurons),
            nn.ReLU(),
            nn.Linear(neurons),
            nn.ReLU(),
        )
        policy_head = nn.Sequential(
            nn.LazyLinear(neurons),
            nn.ReLU(),
            View(*action_space), 
            nn.Softmax(dim=-1)
        )

        value_head = nn.Sequential(
            nn.LazyLinear(neurons),
            nn.ReLU(),
            nn.LazyLinear(infostates),
        )

        return back_bone, value_head, policy_head