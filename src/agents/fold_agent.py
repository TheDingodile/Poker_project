from src.agents.agent import Agent
import torch

class FoldAgent(Agent):
    def __init__(self, bet_sizes: list[float]):
        super(FoldAgent, self).__init__()
        self.bet_sizes = bet_sizes

    def take_action(self, state: torch.Tensor, info: dict[str]) -> str:
        return "f"
    
    def take_action_multiple(self, state: torch.Tensor, info: list[dict[str]]) -> list[str]:
        return ["f" for _ in range(len(info))]
    
    def take_action_PBS(self, state: torch.Tensor, infostates: int) -> torch.Tensor:
        amount_bets = len(self.bet_sizes)
        fold_actions = torch.ones((state.shape[0], infostates, 1))
        call_actions = torch.zeros((state.shape[0], infostates, 1))
        bet_actions = torch.zeros((state.shape[0], infostates, amount_bets)) / amount_bets
        action_not_normalized = torch.cat((call_actions, fold_actions, bet_actions), dim=2)
        return action_not_normalized / action_not_normalized.sum(dim=2, keepdim=True)