import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class RandomAgent(nn.Module):
    def __init__(self, bet_sizes: list[float] = [0.2, 0.5, 1, 2]):
        super(RandomAgent, self).__init__()
        self.bet_sizes = bet_sizes

    def take_action(self, state: dict[str], info: dict[str]) -> str:
        # select random element in action space
        action_space = info["action_space"]
        random_action = random.choice(action_space)
        if not isinstance(random_action, tuple):
            return random_action
        else:
            sizes = [x * state["pot_size"] for x in self.bet_sizes]
            sizes = [x for x in sizes if x >= random_action[0] and x <= random_action[1]]
            sizes.append(random_action[1])
            sizes = [f"b{bet_size:.1f}" for bet_size in sizes]
            return random.choice(sizes)