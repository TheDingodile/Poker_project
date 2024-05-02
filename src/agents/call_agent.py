from src.agents.agent import Agent

class CallAgent(Agent):
    def __init__(self):
        super().__init__()

    def take_action(self, state: dict[str], info: dict[str]):
        return "c"