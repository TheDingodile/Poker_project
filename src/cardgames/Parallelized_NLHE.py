from src.cardgames.NLHE import NLHE
from src.agents.agent import Agent
import torch

class Parallelized_NLHE:
    pass

    def __init__(self, amount_agents: int, stack_depth_bb: int, tables: int) -> None:
        self.amount_agents = amount_agents
        self.games = [NLHE(amount_players=amount_agents, stack_depth_bb=stack_depth_bb) for _ in range(tables)]
        self.states = [None for _ in range(tables)]
        self.rewards =  [None for _ in range(tables)]
        self.dones =  [False for _ in range(tables)]
        self.infos =  [None for _ in range(tables)]
        self.played_hands = 0

    def new_hands(self):
        return [list(result) for result in zip(*(game.new_hand() for game in self.games))]
    
    def step(self, actions):
        for i, is_done in enumerate(self.dones):
            if is_done:
                self.states[i], self.rewards[i], self.dones[i], self.infos[i] = self.games[i].new_hand()
            else:
                self.states[i], self.rewards[i], self.dones[i], self.infos[i] = self.games[i].step(actions[i])

        self.played_hands += sum(self.dones)
        return self.states, self.rewards, self.dones, self.infos
    
    def take_actions(self, states: list[torch.Tensor], infos: list[dict[str]], agents: list[Agent]) -> list[str]:
        batch_states = torch.stack(states)
        idx_of_player_to_act = [game.player_to_act for game in self.games]
        actions = [None for _ in range(len(self.games))]
        for i in range(self.amount_agents):
            agent_to_act_games = [game_number for game_number, idx in enumerate(idx_of_player_to_act) if idx == i and not self.dones[game_number]]
            if len(agent_to_act_games) == 0:
                continue
            
            actions_of_agent = agents[i].take_action_multiple(batch_states[agent_to_act_games], [infos[x] for x in agent_to_act_games])
            for j in range(len(agent_to_act_games)):
                actions[agent_to_act_games[j]] = actions_of_agent[j]
        return actions

    def print_table(self, table_number: int) -> None:
        self.games[table_number].print_table()