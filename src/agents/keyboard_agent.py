from src.agents.agent import Agent

class KeyboardAgent(Agent):
    def __init__(self):
        super().__init__()

    def take_action(self, state):
        action = input("Enter your action: ")
        return action