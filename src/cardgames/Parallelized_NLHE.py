from src.cardgames.NLHE import NLHE
from src.agents.agent import Agent
import torch

class Parallelized_NLHE:
    pass

    def __init__(self, amount_agents: int, stack_depth_bb: int, tables: int) -> None:
        self.amount_agents = amount_agents
        self.games = [NLHE(amount_players=amount_agents, stack_depth_bb=stack_depth_bb) for _ in range(tables)]
        self.states = [None for _ in tables]
        self.rewards =  [None for _ in tables]
        self.dones =  [False for _ in tables]
        self.infos =  [None for _ in tables]

    def new_hands(self):
        return [list(result) for result in zip(*(game.new_hand() for game in self.games))]
    
    def step(self, actions):
        for i, is_done in enumerate(self.dones):
            if is_done:
                self.states[i], self.rewards[i], self.dones[i], self.infos[i] = self.games[i].new_hand()
            else:
                self.states[i], self.rewards[i], self.dones[i], self.infos[i] = self.games[i].step(actions[i])
        return self.states, self.rewards, self.dones, self.infos
    
    def take_actions(self, state: list[dict], info: list[dict], agents: list[Agent]):
        for i in range(len(self.amount_agents)):
            # get all
            agents[i].take_action()

        return [agents[game.player_to_act].take_action(state[i], info[i]) for i, agent in enumerate(agents)]
