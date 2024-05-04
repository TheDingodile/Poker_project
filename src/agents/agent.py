import torch

import torch.nn as nn

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

    def forward(self, state):
        pass

    def take_action(self, state: dict[str], info: dict[str]) -> str:
        pass

    def take_action_PBS(self, state: dict[str], info: dict[str]) -> torch.Tensor:
        pass