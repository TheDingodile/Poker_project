import torch

import torch.nn as nn
import torch.nn.functional as F

class RandomAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(RandomAgent, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, state):
        x = self.fc(state)
        action_probs = F.softmax(x, dim=1)
        action = torch.multinomial(action_probs, num_samples=1)
        return action