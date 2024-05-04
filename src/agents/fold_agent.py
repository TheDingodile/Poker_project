from src.agents.agent import Agent
import torch

class FoldAgent(Agent):
    def __init__(self):
        super(FoldAgent).__init__()

    def take_action(self, state: torch.Tensor, info: dict[str]) -> str:
        return "f"
    
    def take_action_multiple(self, state: torch.Tensor, info: list[dict[str]]) -> list[str]:
        return ["f" for _ in range(len(info))]