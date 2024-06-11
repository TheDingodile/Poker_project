from src.agents.agent import Agent
import torch
import random
import math

class RaiseAgent(Agent):
    def __init__(self, bet_sizes: list[float]):
        super(RaiseAgent, self).__init__()
        self.bet_sizes = bet_sizes

    def take_action(self, state: dict[str], info: dict[str]) -> str:
        random_action = random.choice(self.bet_sizes)
        sizes = [math.ceil(x * info["pot_size"]) for x in self.bet_sizes]
        sizes = [x for x in sizes if x >= random_action[0] and x <= random_action[1]]
        sizes.append(random_action[1])
        sizes = [f"b{bet_size:.1f}" for bet_size in sizes]
        return random.choice(sizes)
    
    def take_action_multiple(self, state: torch.Tensor, info: list[dict[str]]) -> list[str]:
        return [self.take_action(state, info_) for info_ in info]
    
    def take_action_PBS(self, state: torch.Tensor, infostates: int) -> torch.Tensor:
        amount_bets = len(self.bet_sizes)
        fold_actions = torch.zeros((state.shape[0], infostates, 1))
        call_actions = torch.zeros((state.shape[0], infostates, 1))
        bet_actions = torch.ones((state.shape[0], infostates, amount_bets)) / amount_bets
        action_not_normalized = torch.cat((call_actions, fold_actions, bet_actions), dim=2)
        return action_not_normalized / action_not_normalized.sum(dim=2, keepdim=True)
    