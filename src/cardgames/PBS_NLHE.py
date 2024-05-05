from src.cardgames.NLHE import NLHE
from src.cardgames.Parallelized_NLHE import Parallelized_NLHE
from src.cardgames.StandardDeck import Card, Deck
import torch
from src.agents.agent import Agent
import math

class PBS_NLHE:
    def __init__(self, NLHE_games: Parallelized_NLHE, bet_sizes: list[float]) -> None:
        self.bet_sizes = bet_sizes
        self.NLHE_games = NLHE_games
        self.amount_suits = self.NLHE_games.games[0].deck.amount_suits
        self.amount_numbers = self.NLHE_games.games[0].deck.amount_values
        self.amount_players = self.NLHE_games.games[0].amount_players
        self.amount_cards = self.amount_suits * self.amount_numbers
        self.infostates: int = self.amount_cards * (self.amount_cards - 1) // 2
        self.hand_idx_to_infostate_idx_map = self.create_infostate_map()
        self.infostate_idx_to_hand_idx_map = {v: k for k, v in self.hand_idx_to_infostate_idx_map.items()}
        self.public_belief_state = torch.ones(size=(self.NLHE_games.tables, self.infostates, self.amount_players)) / self.infostates

    def create_infostate_map(self) -> dict[int, int]:
        hand_to_infostate_map = {}
        for i in range(self.amount_cards):
            for j in range(i + 1, self.amount_cards):
                hand_to_infostate_map[i * self.amount_cards + j] = len(hand_to_infostate_map)

        for i in range(self.amount_cards):
            for j in range(i):
                hand_to_infostate_map[i * self.amount_cards + j] = hand_to_infostate_map[j * self.amount_cards + i]

        return hand_to_infostate_map
    
    def get_hand_idx(self, hand: list[Card]) -> int:
        return hand[0].id * self.amount_cards + hand[1].id

    def hand_to_infostate_idx(self, hand: list[Card]) -> int:
        return self.hand_idx_to_infostate_idx_map[self.get_hand_idx(hand)]
    
    def infostate_idx_to_hand(self, idx: int) -> list[Card]:
        hand_id = self.infostate_idx_to_hand_idx_map[idx]
        card1_id = hand_id // self.amount_cards
        card2_id = hand_id % self.amount_cards
        return [Card(card1_id % self.amount_suits, card1_id // self.amount_suits + 2), Card(card2_id % self.amount_suits, card2_id // self.amount_suits + 2)]


    def new_hands(self):
        state, _, _, info = self.NLHE_games.new_hands()
        state = torch.stack(state)
        combined_state = torch.cat((self.public_belief_state.flatten(-2), state), dim=1)
        rewards = torch.zeros(size=(self.NLHE_games.tables, self.infostates, self.amount_players))
        dones = torch.zeros(size=(self.NLHE_games.tables, self.infostates))
        return combined_state, rewards, dones, info
    
    def take_actions(self, states: torch.Tensor, agents: list[Agent]) -> torch.Tensor:
        idx_of_player_to_act = [game.player_to_act for game in self.NLHE_games.games]
        actions = [None for _ in range(len(self.NLHE_games.games))]
        for i in range(self.NLHE_games.amount_agents):
            agent_to_act_games = [game_number for game_number, idx in enumerate(idx_of_player_to_act) if idx == i and not self.NLHE_games.dones[game_number]]
            if len(agent_to_act_games) == 0:
                continue
            
            actions_of_agent = agents[i].take_action_PBS(states[agent_to_act_games], self.infostates)
            for j in range(len(agent_to_act_games)):
                actions[agent_to_act_games[j]] = actions_of_agent[j]

        for i, action in enumerate(actions):
            if action is None:
                actions[i] = torch.ones((self.infostates, len(self.bet_sizes) + 2)) / (len(self.bet_sizes) + 2)
        return torch.stack(actions) # torch.Size([tables, infostates, action_space_size])
    
    def get_reward(self, actions: torch.Tensor, idx_of_player_to_act: list[int]) -> torch.Tensor:
        rewards = torch.zeros(size=(self.NLHE_games.tables, self.infostates, self.amount_players))
        table_idxes = torch.arange(self.NLHE_games.tables)

        amount_needed_to_call = torch.tensor([max(NLHE_game.round_pot) - NLHE_game.round_pot[NLHE_game.player_to_act] for NLHE_game in self.NLHE_games.games])
        public_belief_state_of_acting_agent = self.public_belief_state[table_idxes, :, idx_of_player_to_act]
 
        pots_on_tables = torch.tensor([NLHE_game.pot_size for NLHE_game in self.NLHE_games.games])
        bet_sizes = torch.tensor(self.bet_sizes) * pots_on_tables.unsqueeze(1)
        
        c_fractions = actions[:, :, 0] * public_belief_state_of_acting_agent
        bet_fractions = actions[:, :, 2:] * public_belief_state_of_acting_agent.unsqueeze(2)
        rewards[table_idxes, :, idx_of_player_to_act] -= c_fractions * amount_needed_to_call.unsqueeze(1) + torch.sum(bet_fractions * bet_sizes.unsqueeze(1), dim=2)

        return rewards
    
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str]]]:
        # action is shape (tables, infostates, action_space_size)
        idx_of_player_to_act = [game.player_to_act for game in self.NLHE_games.games]
        indices = torch.arange(len(idx_of_player_to_act))

        reward = self.get_reward(actions, idx_of_player_to_act)

        infostate_indexes = []
        for game in self.NLHE_games.games:
            infostate_indexes.append(self.hand_to_infostate_idx(game.hands[game.player_to_act]))
        actions_prob_infostate = actions[indices, infostate_indexes]
        sampled_actions = torch.distributions.Categorical(probs=actions_prob_infostate).sample()
        probabilities_infostates = actions[indices, :, sampled_actions]

        self.public_belief_state[indices, :, idx_of_player_to_act] *= probabilities_infostates
        self.public_belief_state /= self.public_belief_state.sum(dim=1, keepdim=True)
        self.public_belief_state[self.NLHE_games.dones] = 1 / self.infostates

        states, _, _, _ = self.NLHE_games.step(self.action_to_list_of_string(sampled_actions))

        states = torch.stack(states)
        combined_state = torch.cat((self.public_belief_state.flatten(-2), states), dim=1)



        return combined_state, reward, None, None


    def action_to_list_of_string(self, actions: torch.Tensor) -> list[str]:
        actions_str = [None for _ in range(actions.size(0))]
        for i in range(len(actions)):
            if actions[i] == 0:
                actions_str[i] = "c"
            elif actions[i] == 1:
                actions_str[i] = "f"
            else:
                actions_str[i] = f"b{self.bet_sizes[actions[i] - 2]}"
        return actions_str




    def map_to_legal_actions(self, actions: torch.Tensor, infostate_indexes: list[int]) -> torch.Tensor:
        action_spaces = [x["action_space"] for x in self.NLHE_games.infos]
        pot_sizes = torch.tensor([x["pot_size"] for x in self.NLHE_games.infos])
        stack_sizes = torch.tensor([x["stacks"][x["player_to_act"]] for x in self.NLHE_games.infos])
        # find all indexes where action space is only 1 long
        single_action_spaces = [i for i, action_space in enumerate(action_spaces) if len(action_space) == 1]
        min_bets_idx = [i for i, action_space in enumerate(action_spaces) if len(action_space) == 3]
        min_bets_values = [action_spaces[i][0] for i in min_bets_idx]

        actions[single_action_spaces, :, 0] = torch.sum(actions[single_action_spaces, :, 1:], dim=2)
        actions[single_action_spaces, :, 1:] = 0

        for i in range(self.bet_sizes):
            actions[stack_sizes < pot_sizes * self.bet_sizes[i], :, i + 1] = 0


        
    def print_table(self, P0_hide=True, P1_hide=True, game_number=0):
        self.NLHE_games.games[game_number].print_table(P0_hide=P0_hide, P1_hide=P1_hide)