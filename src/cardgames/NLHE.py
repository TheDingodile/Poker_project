from src.cardgames.StandardDeck import Card, Deck
import torch

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
        self.hands = [[] for _ in range(self.amount_players)]
        self.players_in_hand = [False for _ in range(self.amount_players)]
        self.round_pot = [0 for _ in range(self.amount_players)]
        self.refresh_stack = refresh_stack
        self.reward_when_end_of_hand = reward_when_end_of_hand

    def deal_cards(self):
        for _ in range(2):
            for i in range(self.amount_players):
                card = self.deck.deal_card()
                self.hands[i].append(card)

    def next_player(self, player):
        for i in range(player + 1, player + self.amount_players):
            if self.players_in_hand[i % self.amount_players]:
                return i % self.amount_players
        assert False, "No players in hand"
    
    def move_dealer_button(self):
        self.button_position = self.next_player(self.button_position)
        self.player_to_act = self.next_player(self.button_position) if self.amount_players > 2 else self.button_position

    def collect_cards(self):
        self.hands = [[] for _ in range(self.amount_players)]
        self.deck.reset()

    def add_chips(self, amount):
        if amount >= self.stacks[self.player_to_act]:
            amount = self.stacks[self.player_to_act]
        elif amount < 2 * max(self.total_betted_amount_in_hand):
            assert False, "Bet is too small"
        self.stacks[self.player_to_act] -= amount
        self.total_betted_amount_in_hand[self.player_to_act] += amount
        self.round_pot[self.player_to_act] += amount
        self.pot_size += amount
        self.player_to_act = self.next_player(self.player_to_act)

    def gather_blinds(self):
        self.add_chips(0.5)
        self.add_chips(1)

    def new_hand(self):
        self.pot_size = 0
        self.round_pot = [0 for _ in range(self.amount_players)]
        if self.refresh_stack:
            self.stacks = [self.stack_depth_bb for _ in range(self.amount_players)]
        self.players_in_hand = [stack > 0 for stack in self.stacks]
        self.total_betted_amount_in_hand = [0 for _ in range(self.amount_players)]
        self.collect_cards()
        self.deck.shuffle()
        self.move_dealer_button()
        self.gather_blinds()
        self.deal_cards()
        return self.get_state(), 

    def get_reward(self):
        if self.reward_when_end_of_hand:
            if sum(self.players_in_hand) == 1:
                return [self.pot_size if player else -self.total_betted_amount_in_hand for player in self.players_in_hand]
            else:
                return [0 for _ in range(self.amount_players)]
        else:
            if sum(self.players_in_hand) == 1:
                return [self.pot_size if player else -self.total_betted_amount_in_hand for player in self.players_in_hand]

    def is_done(self):
        return sum(self.players_in_hand) == 1 or self.is_showdown()
    
    def is_showdown(self): # Implement this
        return False
    
    def showdown_winner(self): # Implement this
        return torch.argmax(self.players_in_hand)

    def action(self, action):
        if action == "fold":
            pass
        elif action == "check":
            pass
        elif action == "call":
            pass
        elif action == "raise":
            pass
        elif action == "all-in":
            pass





