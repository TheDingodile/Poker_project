import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import math

class RandomNoFoldAgent(nn.Module):
    def __init__(self, bet_sizes: list[float]):
        super(RandomNoFoldAgent, self).__init__()
        self.bet_sizes = bet_sizes

    def take_action_PBS(self, state: torch.Tensor, infostates: int) -> torch.Tensor:
        amount_bets = len(self.bet_sizes)
        non_bet_actions = torch.cat([torch.ones((state.shape[0], infostates, 1)), torch.zeros((state.shape[0], infostates, 1))], dim=2)
        bet_actions = torch.ones((state.shape[0], infostates, amount_bets)) / amount_bets
        action_not_normalized = torch.cat((non_bet_actions, bet_actions), dim=2)
        return action_not_normalized / action_not_normalized.sum(dim=2, keepdim=True)
    