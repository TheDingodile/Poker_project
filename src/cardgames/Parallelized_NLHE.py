from src.cardgames.NLHE import NLHE
from src.agents.agent import Agent
import torch

class Parallelized_NLHE:
    pass

    def __init__(self, amount_agents: int, stack_depth_bb: int, tables: list[NLHE]) -> None:
        self.tables = tables
        self.amount_tables = len(tables)
        self.amount_agents = amount_agents
        self.states = [None for _ in range(self.amount_tables)]
        self.rewards =  [None for _ in range(self.amount_tables)]
        self.dones =  [False for _ in range(self.amount_tables)]
        self.infos =  [None for _ in range(self.amount_tables)]
        self.played_hands = 0

    def new_hands(self) -> tuple[list[torch.Tensor], list[list[float]], list[bool], list[dict[str]]]:
        return [list(result) for result in zip(*(game.new_hand() for game in self.tables))]
    
    def step(self, actions: list[str]) -> tuple[list[torch.Tensor], list[list[float]], list[bool], list[dict[str]]]:
        for i, is_done in enumerate(self.dones):
            self.states[i], self.rewards[i], self.dones[i], self.infos[i] = self.tables[i].step(actions[i])
            if self.dones[i]:
                self.tables[i].new_hand()

        self.played_hands += sum(self.dones)
        return self.states, self.rewards, self.dones, self.infos
    
    def take_actions(self, states: list[torch.Tensor], infos: list[dict[str]], agents: list[Agent]) -> list[str]:
        batch_states = torch.stack(states)
        idx_of_player_to_act = [game.player_to_act for game in self.tables]
        actions = [None for _ in range(self.amount_tables)]
        for i in range(self.amount_agents):
            agent_to_act_games = [game_number for game_number, idx in enumerate(idx_of_player_to_act) if idx == i]
            if len(agent_to_act_games) == 0:
                continue
            
            actions_of_agent = agents[i].take_action_multiple(batch_states[agent_to_act_games], [infos[x] for x in agent_to_act_games])
            for j in range(len(agent_to_act_games)):
                actions[agent_to_act_games[j]] = actions_of_agent[j]
        return actions

    def print_table(self, table_number: int) -> None:
        self.tables[table_number].print_table()