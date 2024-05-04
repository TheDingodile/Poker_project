import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import math

class RandomAgent(nn.Module):
    def __init__(self, bet_sizes: list[float] = [0.1, 0.2, 0.5, 1, 2, 5]):
        super(RandomAgent, self).__init__()
        self.bet_sizes = bet_sizes

    def take_action(self, state: dict[str], info: dict[str]) -> str:
        random_action = random.choice(info["action_space"])
        if not isinstance(random_action, tuple):
            return random_action
        else:
            # select random bet size, round up to nearest integer
            sizes = [math.ceil(x * info["pot_size"]) for x in self.bet_sizes]
            sizes = [x for x in sizes if x >= random_action[0] and x <= random_action[1]]
            sizes.append(random_action[1])
            sizes = [f"b{bet_size:.1f}" for bet_size in sizes]
            return random.choice(sizes)
        
    def take_action_multiple(self, state: torch.Tensor, info: list[dict[str]]) -> list[str]:
        return [self.take_action(state, info_) for info_ in info]