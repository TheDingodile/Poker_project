from src.agents.agent import Agent

class FoldAgent(Agent):
    def __init__(self):
        super(FoldAgent).__init__()

    def take_action(self, state: dict[str], info: dict[str]):
        return "f"