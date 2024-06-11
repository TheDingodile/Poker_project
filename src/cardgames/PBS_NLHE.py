from src.cardgames.NLHE import NLHE
from src.cardgames.Parallelized_NLHE import Parallelized_NLHE
from src.cardgames.StandardDeck import Card, Deck
import torch
from src.agents.agent import Agent
import math
import time
from phevaluator.evaluator import evaluate_cards


class PBS_NLHE:
    def __init__(self, NLHE_games: Parallelized_NLHE, bet_sizes: list[float]) -> None:
        self.bet_sizes = bet_sizes
        self.NLHE_games = NLHE_games
        self.amount_suits = self.NLHE_games.tables[0].deck.amount_suits
        self.amount_numbers = self.NLHE_games.tables[0].deck.amount_values
        self.amount_players = self.NLHE_games.tables[0].amount_players
        self.amount_cards = self.amount_suits * self.amount_numbers
        self.cards_on_hand = self.NLHE_games.tables[0].cards_on_hand
        self.hand_to_infostate_map = {}
        self.create_hand_to_infostate_map(self.cards_on_hand, 0, [])
        self.amount_infostates: int = len(self.hand_to_infostate_map)
        self.infostate_to_hand_map = {v: k for k, v in self.hand_to_infostate_map.items()}
        self.public_belief_state = torch.ones(size=(self.NLHE_games.amount_tables, self.amount_infostates, self.amount_players)) / self.amount_infostates
        self.card_to_impossible_infostate = self.create_card_to_impossible_infostate()
        self.hand_to_impossible_infostate = self.create_hand_to_impossible_infostate()
        self.infostate_matrix = self.create_infostate_to_impossible_infostate_matrix()
        self.all_hands = self.make_all_hands_list()
        self.total_reward = [0, 0]

    def make_all_hands_list(self) -> list[str]:
        all_hands = []
        for i, hand in self.infostate_to_hand_map.items():
            all_hands.append(tuple([Card(hand[j] % self.amount_suits, hand[j] // self.amount_suits + 2, self.amount_numbers, self.amount_suits).to_string() for j in range(self.cards_on_hand)]))
        return all_hands

    def create_infostate_to_impossible_infostate_matrix(self) -> torch.Tensor:
        infostate_to_possible_infostates = torch.ones(1, self.amount_infostates, self.amount_infostates)
        for hand in self.hand_to_impossible_infostate:
            for impossible_infostate in self.hand_to_impossible_infostate[hand]:
                infostate_to_possible_infostates[:, hand, impossible_infostate] = 0
        return infostate_to_possible_infostates
    
    def create_card_to_impossible_infostate(self) -> dict[int, list[int]]:
        card_to_impossible_infostate = {}
        for card_id in range(self.amount_cards):
            impossible_infostates = []
            for infostate_id in range(self.amount_infostates):
                if card_id in self.infostate_to_hand_map[infostate_id]:
                    impossible_infostates.append(infostate_id)
            card_to_impossible_infostate[card_id] = impossible_infostates
        return card_to_impossible_infostate
    
    def create_hand_to_impossible_infostate(self) -> dict[tuple[int], list[int]]:
        hand_to_impossible_infostate = {}
        for hand, infostate_idx in self.hand_to_infostate_map.items():
            impossible_infostates = set()
            for card_id in hand:
                impossible_infostates = impossible_infostates.union(set(self.card_to_impossible_infostate[card_id]))
            hand_to_impossible_infostate[infostate_idx] = list(impossible_infostates)
        return hand_to_impossible_infostate

    def create_hand_to_infostate_map(self, depth_left:int, start:int, hand:list[int]) -> None:
        if depth_left == 0:
            self.hand_to_infostate_map[tuple(hand)] = len(self.hand_to_infostate_map)
            return
        for i in range(start, self.amount_cards):
            hand.append(i)
            self.create_hand_to_infostate_map(depth_left - 1, i + 1, hand)
            hand.pop()

    def hand_to_infostate_idx(self, hand: list[Card]) -> int:
        return self.hand_to_infostate_map[tuple(sorted([card.id for card in hand]))]
    

    def infostate_to_hand(self, idx: int) -> list[Card]:
        hand_as_tuple = self.infostate_to_hand_map[idx]
        hand = []
        for card_id in hand_as_tuple:
            suit = card_id % self.amount_suits
            number = card_id // self.amount_suits + 2
            hand.append(Card(suit, number, self.amount_numbers, self.amount_suits))
        return hand


    def new_hands(self):
        state, _, _, info = self.NLHE_games.new_hands()
        state = torch.stack(state)
        combined_state = torch.cat((self.public_belief_state.flatten(-2), state), dim=1)
        rewards = torch.zeros(size=(self.NLHE_games.amount_tables, self.amount_infostates, self.amount_players))
        dones = torch.zeros(size=(self.NLHE_games.amount_tables, self.amount_infostates))
        return combined_state, rewards, dones, info
    
    def take_actions(self, states: torch.Tensor, agents: list[Agent]) -> torch.Tensor:
        idx_of_player_to_act = [game.player_to_act for game in self.NLHE_games.tables]
        actions = [None for _ in range(self.NLHE_games.amount_tables)]
        for i in range(self.NLHE_games.amount_agents):
            agent_to_act_games = [game_number for game_number, idx in enumerate(idx_of_player_to_act) if idx == i]
            if len(agent_to_act_games) == 0:
                continue
            
            actions_of_agent = agents[i].take_action_PBS(states[agent_to_act_games], self.amount_infostates)
            for j in range(len(agent_to_act_games)):
                actions[agent_to_act_games[j]] = actions_of_agent[j]

        for i, action in enumerate(actions):
            if action is None:
                actions[i] = torch.ones((self.amount_infostates, len(self.bet_sizes) + 2)) / (len(self.bet_sizes) + 2)
        return torch.stack(actions) # torch.Size([tables, infostates, action_space_size])
    
    def get_reward_non_showdown(self, actions: torch.Tensor, idx_of_player_to_act: list[int]) -> torch.Tensor:
        # fold actions is only implements for heads-up so far
        rewards = torch.zeros(size=(self.NLHE_games.amount_tables, self.amount_infostates, self.amount_players))
        table_idxes = torch.arange(self.NLHE_games.amount_tables)

        amount_needed_to_call = torch.tensor([min(max(NLHE_game.round_pot) - NLHE_game.round_pot[NLHE_game.player_to_act], NLHE_game.stacks[NLHE_game.player_to_act]) for NLHE_game in self.NLHE_games.tables])
        public_belief_state_of_acting_agent = self.public_belief_state[table_idxes, :, idx_of_player_to_act]
        pots_on_tables = torch.tensor([NLHE_game.pot_size for NLHE_game in self.NLHE_games.tables])
        bet_sizes = (torch.tensor(self.bet_sizes) * pots_on_tables.unsqueeze(1))
        stack_of_acting_player = torch.tensor([NLHE_game.stacks[NLHE_game.player_to_act] for NLHE_game in self.NLHE_games.tables])
        print(stack_of_acting_player.shape)
        print(bet_sizes.shape)
        bet_sizes[bet_sizes > stack_of_acting_player.unsqueeze(1)] = stack_of_acting_player

        # need to do something here to only bet all_in_maximally if the stack is smaller than the pot
        # bet_sizes[bet_sizes > self.NLHE_games.tables[0].stack_depth_bb] = self.NLHE_games.tables[0].stack_depth_bb
        
        c_fractions = actions[:, :, 0] * public_belief_state_of_acting_agent
        bet_fractions = actions[:, :, 2:] * public_belief_state_of_acting_agent.unsqueeze(2)
        
        rewards[table_idxes, :, idx_of_player_to_act] -= c_fractions * amount_needed_to_call.unsqueeze(1) + torch.sum(bet_fractions * bet_sizes.unsqueeze(1), dim=2)

        # this part is specific to heads-up
        f_fractions = actions[:, :, 1] * public_belief_state_of_acting_agent
        rewards[table_idxes, :, [1 - x for x in idx_of_player_to_act]] += f_fractions * pots_on_tables.unsqueeze(1)
        return rewards
    
    def get_showdown_rewards(self, infos: list[dict[str]]) -> torch.Tensor:
        # print(infos)
        rewards = torch.zeros(size=(self.NLHE_games.amount_tables, self.amount_infostates, self.amount_players))
        show_idxes = [info["is_showdown"] for info in infos]
        idx_of_showdown_tables = [i for i, show_idx in enumerate(show_idxes) if show_idx]
        if len(idx_of_showdown_tables) > 0:
            equity_per_infostate_per_player = self.get_equity_of_players(idx_of_showdown_tables, infos) 
            for i, table_idx in enumerate(idx_of_showdown_tables):
                rewards[table_idx] += equity_per_infostate_per_player[i] * self.public_belief_state[table_idx] * infos[table_idx]["pot_size"]
                
        return rewards
    
    def get_showdown_ranking(self, idx_of_showdown_tables: list[int], infos: list[dict[str]]) -> torch.Tensor:
        amount_of_showdown_tables = len(idx_of_showdown_tables)
        ranking = torch.zeros(amount_of_showdown_tables, self.amount_infostates)
        for i, table_idx in enumerate(idx_of_showdown_tables):
            for j, hand in enumerate(self.all_hands):
                # check if the hand is possible
                if self.public_belief_state[table_idx, j].sum() == 0:
                    continue
                try:
                    ranking[i, j] = evaluate_cards(*[card.to_string() for card in infos[table_idx]['community_cards']], *hand)
                except:
                    print("public belief state", self.public_belief_state[table_idx])
                    raise ValueError(f"Error in evaluate_cards with community cards {[card.to_string() for card in infos[table_idx]['community_cards']]} and hand {hand}")
        # add small random noise to make sure that the ranking is unique
        ranking += torch.rand_like(ranking)
        ranking = ranking.argsort(descending=True)
        return ranking
    
    def get_equity_of_players(self, idx_of_showdown_tables: list[int], infos: list[dict[str]]) -> torch.Tensor:
        ranking = self.get_showdown_ranking(idx_of_showdown_tables, infos)
        equity = torch.empty(len(idx_of_showdown_tables), self.amount_infostates, self.amount_players)
        for player in range(self.amount_players):
            # multiply the PBS with the infostate_to_possible_infostates to get the infostate matrix, unsqueeze to make it work
            infostate_matrix = self.public_belief_state[idx_of_showdown_tables,:,player:player + 1].transpose(2, 1) * self.infostate_matrix
            # print(ranking.shape, infostate_matrix.shape)
            expanded_permutation = ranking.unsqueeze(1).expand(infostate_matrix.shape)

            infostate_matrix_permuted = torch.gather(infostate_matrix, 2, expanded_permutation)
            cum_sum_info_matrix = torch.cumsum(infostate_matrix_permuted, dim=2)
            argsorted_permutation = torch.argsort(ranking)

            equity_per_infostate_unnormalized = torch.gather(cum_sum_info_matrix, 2, argsorted_permutation.unsqueeze(2)).squeeze(2)
            equity_per_infostate = equity_per_infostate_unnormalized / cum_sum_info_matrix[:, :, -1]
            equity[:, :, 1 - player] = equity_per_infostate
        return equity


    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str]]]:
        # action is shape (tables, infostates, action_space_size)
        idx_of_player_to_act = [game.player_to_act for game in self.NLHE_games.tables]
        indices = torch.arange(len(idx_of_player_to_act))

        reward = self.get_reward_non_showdown(actions, idx_of_player_to_act)


        # print(self.public_belief_state)
        infostate_indexes = []
        for game in self.NLHE_games.tables:
            infostate_indexes.append(self.hand_to_infostate_idx(game.hands[game.player_to_act]))
        actions_prob_infostate = actions[indices, infostate_indexes]

        sampled_actions = torch.distributions.Categorical(probs=actions_prob_infostate).sample()
        probabilities_infostates = actions[indices, :, sampled_actions]
        states, _, _, infos = self.NLHE_games.step(self.action_to_list_of_string(sampled_actions))
        self.total_reward = [self.total_reward[0] + torch.sum(torch.tensor(self.NLHE_games.rewards)[:, 0]), self.total_reward[1] + torch.sum(torch.tensor(self.NLHE_games.rewards)[:, 1])]
        print(self.total_reward, "real reward")
        # this part removes community cards from the public belief state
        for i, game in enumerate(self.NLHE_games.tables):
            if game.cards_of_this_round_community_cards is None:
                continue
            for card in game.cards_of_this_round_community_cards:
                self.public_belief_state[i, self.card_to_impossible_infostate[card.id]] = 0
        # this part updates the public belief state

        self.public_belief_state[indices, :, idx_of_player_to_act] *= probabilities_infostates
        self.public_belief_state /= self.public_belief_state.sum(dim=1, keepdim=True)
        reward += self.get_showdown_rewards(infos)
        self.public_belief_state[self.NLHE_games.dones] = 1 / self.amount_infostates

        states = torch.stack(states)
        combined_state = torch.cat((self.public_belief_state.flatten(-2), states), dim=1)


        return combined_state, reward, self.NLHE_games.dones, self.NLHE_games.infos


    def action_to_list_of_string(self, actions: torch.Tensor) -> list[str]:
        actions_str = [None for _ in range(actions.size(0))]
        for i in range(len(actions)):
            if actions[i] == 0:
                actions_str[i] = "c"
            elif actions[i] == 1:
                actions_str[i] = "f"
            else:
                actions_str[i] = f"b{self.bet_sizes[actions[i] - 2] * self.NLHE_games.tables[i].pot_size:.1f}"
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
        self.NLHE_games.tables[game_number].print_table(P0_hide=P0_hide, P1_hide=P1_hide)