from src.agents.agent import Agent

class KeyboardAgent(Agent):
    def __init__(self):
        super(KeyboardAgent).__init__()

    def take_action(self, state: dict[str], info: dict[str]):
        action = input("Enter your action: ")
        return action