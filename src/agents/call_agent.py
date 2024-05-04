from src.agents.agent import Agent
import torch

class CallAgent(Agent):
    def __init__(self):
        super().__init__()

    def take_action(self, state: torch.Tensor, info: dict[str]) -> str:
        return "c"
    
    def take_action_multiple(self, state: torch.Tensor, info: list[dict[str]]) -> list[str]:
        return ["c" for _ in range(len(info))]