from src.cardgames.StandardDeck import Card, Deck
import torch
from phevaluator.evaluator import evaluate_cards

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def receive_card(self, card):
        self.hand.append(card)

    def show_hand(self):
        for card in self.hand:
            print(f"{card}", end=" ")
        print()

class NLHE:
    def __init__(self, amount_players, stack_depth_bb=100, refresh_stack=True, reward_when_end_of_hand=True):
        self.deck = Deck()
        self.amount_players = amount_players
        self.button_position = 0
        self.player_to_act = None
        self.pot_size = 0
        self.stack_depth_bb = stack_depth_bb
        self.stacks = [stack_depth_bb for _ in range(self.amount_players)]
        self.total_betted_amount_in_hand = [0 for _ in range(self.amount_players)]
        self.hands: list[list[Card | None]] = [[None, None] for _ in range(self.amount_players)]
        self.players_in_hand = [False for _ in range(self.amount_players)]
        self.round_pot = [None for _ in range(self.amount_players)]
        self.refresh_stack = refresh_stack
        self.reward_when_end_of_hand = reward_when_end_of_hand
        self.community_cards = [None, None, None, None, None]
        self.history = []

    def deal_cards(self):
        for j in range(2):
            for i in range(self.amount_players):
                card = self.deck.deal_card()
                self.hands[i][j](card)

    def next_player(self, player):
        for i in range(player + 1, player + self.amount_players):
            if self.players_in_hand[i % self.amount_players]:
                return i % self.amount_players
        assert False, "No players in hand"
    
    def move_dealer_button(self):
        self.button_position = self.next_player(self.button_position)
        self.player_to_act = self.next_player(self.button_position) if self.amount_players > 2 else self.button_position

    def collect_cards(self):
        self.hands = [[None, None] for _ in range(self.amount_players)]
        self.deck.reset()

    def add_chips(self, amount):
        if amount >= self.stacks[self.player_to_act]:
            amount = self.stacks[self.player_to_act]
        elif amount < 2 * max(self.total_betted_amount_in_hand):
            assert False, "Bet is too small"
        self.stacks[self.player_to_act] -= amount
        self.total_betted_amount_in_hand[self.player_to_act] += amount
        if self.round_pot[self.player_to_act] is None:
            self.round_pot[self.player_to_act] = 0
        self.round_pot[self.player_to_act] += amount
        self.pot_size += amount
        self.player_to_act = self.next_player(self.player_to_act)

    def gather_blinds(self):
        self.add_chips(0.5)
        self.add_chips(1)

    def new_hand(self):
        self.pot_size = 0
        self.round_pot = [None for _ in range(self.amount_players)]
        if self.refresh_stack:
            self.stacks = [self.stack_depth_bb for _ in range(self.amount_players)]
        self.players_in_hand = [stack > 0 for stack in self.stacks]
        self.total_betted_amount_in_hand = [0 for _ in range(self.amount_players)]
        self.collect_cards()
        self.deck.shuffle()
        self.move_dealer_button()
        self.gather_blinds()
        self.deal_cards()
        return self.get_state(), [0 for _ in range(self.amount_players)], False, {}

    def get_reward(self):
        if self.reward_when_end_of_hand:
            if sum(self.players_in_hand) == 1:
                return [self.pot_size if player else -self.total_betted_amount_in_hand for player in self.players_in_hand]
            elif self.is_showdown():
                return self.get_showdown_reward()
            else:
                return [0 for _ in range(self.amount_players)]
        else:
            if sum(self.players_in_hand) == 1:
                return [self.pot_size if player else 0 for player in self.players_in_hand]
            else:
                # Implement this
                pass

    def is_done(self):
        return sum(self.players_in_hand) == 1 or self.is_showdown()
    
    def is_showdown(self):
        return any(self.community_cards == None) == False and self.is_next_round()
    
    def is_next_round(self):
        if any([self.round_pot[i] for i in range(self.amount_players) if self.players_in_hand[i] and self.stacks[i] > 0]) is None:
            return False
        if len(set([self.round_pot[i] for i in range(self.amount_players) if self.players_in_hand[i] and self.stacks[i] > 0])) != 1:
            return False
        return True 

    def step(self, action: str):
        assert action in ["fold", "call"] or action[:4] == "bet:"
        if action == "fold":
            self.players_in_hand[self.player_to_act] = False
            self.player_to_act = self.next_player(self.player_to_act)
        elif action == "call":
            self.add_chips(max(self.total_betted_amount_in_hand) - self.total_betted_amount_in_hand[self.player_to_act])
        else:
            amount = int(action[4:])
            self.add_chips(amount)
        if self.is_done():
            return self.get_state(), self.get_reward(), True, {}
        
    def get_state(self):
        return {
            "player_to_act": self.player_to_act,
            "stacks": self.stacks,
            "total_betted_amount_in_hand": self.total_betted_amount_in_hand,
            "hands": self.hands,
            "community_cards": self.community_cards,
            "players_in_hand": self.players_in_hand,
            "round_pot": self.round_pot,
            "pot_size": self.pot_size
        }

    
    def check_showdown_ranking(self):
        if not self.is_showdown():
            assert False, "No showdown"

        hand_strengths = [torch.inf for _ in range(self.amount_players)]
        for i in range(self.amount_players):
            if self.players_in_hand[i]:
                hand_strength = evaluate_cards([card.to_string() for card in self.hands[i] + self.community_cards])
                hand_strengths[i] = hand_strength
        return torch.argsort(hand_strengths)
    
    def get_showdown_reward(self):
        ranking = self.check_showdown_ranking()
        # filter out players that are not in hand
        ranking = [player for player in ranking if self.players_in_hand[player]]
        rewards = [0 for _ in range(self.amount_players)]
        contributions_to_pot = self.total_betted_amount_in_hand.copy()
        for i in range(len(ranking)):
            my_contribution = contributions_to_pot[ranking[i]]
            contributions_to_pot_for_me = [min(my_contribution, x) for x in contributions_to_pot]
            rewards[ranking[i]] = sum(contributions_to_pot_for_me)
            for j in range(self.amount_players):
                contributions_to_pot[j] -= min(contributions_to_pot_for_me[j], contributions_to_pot[j])
        return rewards




