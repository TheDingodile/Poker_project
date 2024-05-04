from src.agents.agent import Agent
import torch

class KeyboardAgent(Agent):
    def __init__(self):
        super(KeyboardAgent).__init__()

    def take_action(self, state: dict[str], info: dict[str]):
        action = input("Enter your action: ")
        return action
    
    def take_action_multiple(self, state: torch.Tensor, info: list[dict[str]]):
        action = input("Enter your action: ")
        return [action for _ in range(len(info))]