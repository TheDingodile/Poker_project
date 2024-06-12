from src.cardgames.StandardDeck import Card, Deck
import torch
from phevaluator.evaluator import evaluate_cards
from src.utils import argsorted_index_lists
import math

class NLHE:
    def __init__(self, amount_players: int, stack_depth_bb: int, amount_values: int, amount_suits: int, cards_on_hand: int, amount_community_cards: int, refresh_stack=True, reward_when_end_of_hand=True):
        self.deck = Deck(amount_values, amount_suits)
        self.amount_players = amount_players
        self.amount_community_cards = amount_community_cards
        self.button_position: int = 0
        self.player_to_act: int = 0
        self.pot_size: float = 0.0
        self.stack_depth_bb = stack_depth_bb
        self.cards_on_hand = cards_on_hand
        self.stacks = [stack_depth_bb for _ in range(self.amount_players)]
        self.total_betted_amount_in_hand: list[float] = [0.0 for _ in range(self.amount_players)]
        self.hands: list[list[Card | None]] = [[None] * cards_on_hand for _ in range(self.amount_players)]
        self.players_in_hand = [False for _ in range(self.amount_players)]
        self.round_pot = [0.0 for _ in range(self.amount_players)]
        self.had_chance_to_act = [False for _ in range(self.amount_players)]
        self.refresh_stack = refresh_stack
        self.reward_when_end_of_hand = reward_when_end_of_hand
        self.community_cards: list[Card] = [None] * self.amount_community_cards
        self.history = []
        self.total_earnings = [0.0 for _ in range(self.amount_players)]
        self.cards_of_this_round_community_cards: list[Card] = []


    def deal_cards(self):
        for j in range(self.cards_on_hand):
            for i in range(self.amount_players):
                card = self.deck.deal_card()
                self.hands[i][j] = card
        # self.hands = [sorted(player, key=lambda x: x.id, reverse=True) for player in self.hands]

    def next_player(self, player):
        for i in range(player + 1, player + self.amount_players):
            if self.players_in_hand[i % self.amount_players]:
                return i % self.amount_players
        assert False, "No players in hand"
    
    def move_dealer_button(self):
        self.button_position = self.next_player(self.button_position)
        self.player_to_act = self.next_player(self.button_position) if self.amount_players > 2 else self.button_position

    def collect_cards(self):
        self.hands = [[None] * self.cards_on_hand for _ in range(self.amount_players)]
        self.community_cards = [None] * self.amount_community_cards
        self.deck.reset()

    def round_nearest_sb(self, amount):
        return math.ceil(amount * 2) / 2

    def get_min_bet(self):
        sorted_bet_sizes = sorted(self.round_pot)
        last_raise = sorted_bet_sizes[-1] - sorted_bet_sizes[-2]
        to_call = sorted_bet_sizes[-1] - self.round_pot[self.player_to_act]
        return self.round_nearest_sb(max(1, last_raise) + to_call)

    def add_chips(self, amount, is_call=False, blinds=False):
        amount = self.round_nearest_sb(amount)
        if (amount + 0.5) >= self.stacks[self.player_to_act]:
            amount = self.stacks[self.player_to_act]
        elif not blinds and not is_call and amount < self.get_min_bet():
            assert False, f"Bet of {amount} has to by minimum {self.get_min_bet()} and so is too small"
        self.stacks[self.player_to_act] -= amount
        self.total_betted_amount_in_hand[self.player_to_act] += amount
        self.round_pot[self.player_to_act] += amount
        self.pot_size += amount
        self.player_to_act = self.next_player(self.player_to_act)

    def gather_blinds(self):
        self.add_chips(0.5, blinds=True)
        self.add_chips(1.0, blinds=True)

    def new_hand(self):
        self.pot_size = 0
        self.round_pot = [0.0 for _ in range(self.amount_players)]
        if self.refresh_stack:
            self.stacks = [self.stack_depth_bb for _ in range(self.amount_players)]
        self.players_in_hand = [stack > 0.0 for stack in self.stacks]
        self.total_betted_amount_in_hand = [0.0 for _ in range(self.amount_players)]
        self.had_chance_to_act = [False for _ in range(self.amount_players)]
        self.collect_cards()
        self.deck.shuffle()
        self.move_dealer_button()
        self.gather_blinds()
        self.deal_cards()
        return self.get_state(), [0.0 for _ in range(self.amount_players)], False, self.get_info()

    def get_reward(self, is_showdown=False):
        if self.reward_when_end_of_hand:
            if self.one_player_left():
                return [self.pot_size - self.total_betted_amount_in_hand[i] if player else -self.total_betted_amount_in_hand[i] for i, player in enumerate(self.players_in_hand)]
            elif is_showdown:
                pot_distribution = self.distribute_pot()
                reward = [x - y for x, y in zip(pot_distribution, self.total_betted_amount_in_hand)]
                return reward
            else:
                return [0.0 for _ in range(self.amount_players)]
        else:
            if self.one_player_left():
                return [self.pot_size - self.total_betted_amount_in_hand[i] if player else -self.total_betted_amount_in_hand[i] for i, player in enumerate(self.players_in_hand)]
            else:
                # Implement this
                pass

    def is_showdown(self):
        return self.is_river() and self.is_next_round() and not self.one_player_left()

    def one_player_left(self):
        return sum(self.players_in_hand) == 1
    
    def is_river(self):
        return all([card is not None for card in self.community_cards])
    
    def is_next_round(self):
        if any([not self.had_chance_to_act[i] for i in range(self.amount_players) if self.players_in_hand[i] and self.stacks[i] > 0]):
            return False
        if all([self.round_pot[i] == max(self.round_pot) for i in range(self.amount_players) if self.players_in_hand[i] and self.stacks[i] > 0]):
            return True
        return False 

    def step(self, action: str, print_action_legality=False, keep_community_cards=False):
        # check action legality
        if action not in ["f", "c"] and (len(action) == 0 or action[0] != "b"):
            if print_action_legality:
                print("action not understood, taking call action")
                print("action space is", self.get_action_space())
            return self.step(action="c")
        elif action[0] == "b":
            try:
                amount = float(action[1:])
                if amount < self.get_min_bet() and amount != self.stacks[self.player_to_act]:
                    if print_action_legality:
                        print("betted amount too small, taking call action")
                        print("action space is", self.get_action_space())
                    return self.step(action="c")
            except:
                if print_action_legality:
                    print("betted amount not understood, taking call action")
                    print("action space is", self.get_action_space())
                    print("your bet was", action)
                return self.step(action="c")
        if not keep_community_cards:
            self.cards_of_this_round_community_cards = [] # reset this variable
        self.had_chance_to_act[self.player_to_act] = True
        if action == "f":
            self.players_in_hand[self.player_to_act] = False
            self.player_to_act = self.next_player(self.player_to_act)
        elif action == "c":
            self.add_chips(max(self.total_betted_amount_in_hand) - self.total_betted_amount_in_hand[self.player_to_act], is_call=True)
        else:
            amount = float(action[1:])
            self.add_chips(amount)

        all_folded = self.one_player_left()
        is_river = self.is_river()
        go_to_next_round = self.is_next_round()

        # check if game is done
        if all_folded or (is_river and go_to_next_round):
            reward = self.get_reward(is_showdown=not all_folded)
            self.total_earnings = [sum(x) for x in zip(self.total_earnings, reward)]
            return self.get_state(), reward, True, self.get_info()

        if go_to_next_round:
            # make self.player_to_act first person after button that is still in hand
            self.player_to_act = self.next_player(self.button_position)
            self.round_pot = [0.0 for _ in range(self.amount_players)]
            self.had_chance_to_act = [False for _ in range(self.amount_players)]
            self.deal_community_cards()


        if self.stacks[self.player_to_act] == 0:
            return self.step(action="c", keep_community_cards=True)

        return self.get_state(), self.get_reward(is_showdown=False), False, self.get_info()
    
    def deal_community_cards(self):

        if self.amount_community_cards < 3:
            cards = [self.deck.deal_card() for _ in range(self.amount_community_cards)]
            self.community_cards = cards
        else:
            if self.community_cards[0] is None:
                cards = [self.deck.deal_card() for _ in range(3)]
                self.community_cards[:3] = cards
            elif self.community_cards[-1] is None:
                for index, element in enumerate(self.community_cards):
                    if element is None:
                        cards = [self.deck.deal_card()]
                        self.community_cards[index: index + 1] = cards
                        break
            else:
                assert False, "All community cards are already dealt"
        self.cards_of_this_round_community_cards.extend(cards)

        
    def get_state(self):
        # pot_size to stack, round_pot to pot_size, who to act, who is button, community_cards
        player_features = self.amount_players * 2
        table_features = 3
        community_card_features = self.deck.amount_values * self.deck.amount_suits
        state = torch.zeros(size=(player_features + table_features + community_card_features,), dtype=torch.float)
        state[0] = self.player_to_act
        state[1] = self.button_position
        state[2] = self.pot_size
        for i in range(self.amount_players):
            state[i * 2 + table_features] = self.pot_size / (self.stacks[i] + self.pot_size)
            state[i * 2 + table_features + 1] = self.round_pot[i] / self.pot_size
        for i, card in enumerate(self.community_cards):
            if card is not None:
                state[player_features + table_features + card.id] = 1.0
        return state
    
    def get_info(self):
        return {
            "player_to_act": self.player_to_act,
            "stacks": self.stacks,
            "total_betted_amount_in_hand": self.total_betted_amount_in_hand,
            "hand": self.hands[self.player_to_act],
            "community_cards": self.community_cards,
            "players_in_hand": self.players_in_hand,
            "round_pot": self.round_pot,
            "pot_size": self.pot_size,
            "button_position": self.button_position,
            "hands": self.hands,
            "action_space": self.get_action_space(),
            "is_showdown": self.is_showdown(),
        }
    

    
    def get_action_space(self):
        action_space = ["c"]
        if self.stacks[self.player_to_act] > 0:
            action_space.append("f")
        min_bet = self.get_min_bet()
        if self.stacks[self.player_to_act] >= min_bet:
            action_space.append((max(1.0, min_bet), self.stacks[self.player_to_act]))
        elif self.stacks[self.player_to_act] > 0:
            action_space.append((self.stacks[self.player_to_act], self.stacks[self.player_to_act]))
        return action_space

    def check_showdown_ranking(self):
        hand_strengths = [torch.inf for _ in range(self.amount_players)]
        for i in range(self.amount_players):
            if self.players_in_hand[i]:
                hand_strength = evaluate_cards(*[card.to_string() for card in self.hands[i] + self.community_cards])
                hand_strengths[i] = hand_strength
        # argsorted = [x for x, y in sorted(enumerate(hand_strengths), key = lambda x: x[1])]
        return argsorted_index_lists(hand_strengths)
    
    def distribute_pot(self):
        ranking = self.check_showdown_ranking()
        # filter out players that are not in hand
        # ranking = [player for player in ranking if self.players_in_hand[player]]

        rewards = [0 for _ in range(self.amount_players)]
        contributions_to_pot = self.total_betted_amount_in_hand.copy()

        for i in range(len(ranking)):
            ranking[i] = [x for x in ranking[i] if contributions_to_pot[x] > 0]
            if not ranking[i]:
                continue
            our_contributions_to_pot = [contributions_to_pot[x] for x in ranking[i]]
            our_max_contribution_to_pot = max(our_contributions_to_pot)
            all_contribution_to_our_pot = [min(our_max_contribution_to_pot, x) for x in contributions_to_pot]
            our_contributions_to_pot_total = sum(our_contributions_to_pot)
            our_ratio_of_pot = [x/our_contributions_to_pot_total for x in our_contributions_to_pot]
            our_pot = sum(all_contribution_to_our_pot)
            for k, winner in enumerate(ranking[i]):
                rewards[winner] = our_pot * our_ratio_of_pot[k]
                
            for j in range(self.amount_players):
                contributions_to_pot[j] -= min(all_contribution_to_our_pot[j], contributions_to_pot[j])

        # for i in range(len(ranking)):
        #     my_contribution = contributions_to_pot[ranking[i]]
        #     contributions_to_pot_for_me = [min(my_contribution, x) for x in contributions_to_pot]
        #     rewards[ranking[i]] = sum(contributions_to_pot_for_me)
        #     for j in range(self.amount_players):
        #         contributions_to_pot[j] -= min(contributions_to_pot_for_me[j], contributions_to_pot[j])
        return rewards


    def print_table(self, P0_hide=False, P1_hide=False):
        # print a small image with information about the game

        community_cards = [x.__repr__() for x in self.community_cards if x is not None]
        community_cards_str = ' '.join(community_cards)
        community_cards_length = len(community_cards_str)
        spaces_needed = (22 - community_cards_length) // 2
        community_cards_display = " " * spaces_needed + community_cards_str + " " * (spaces_needed + community_cards_length % 2)

        pot_size_str = f"{(self.pot_size - sum(self.round_pot)):.1f}bb"
        pot_size_length = len(pot_size_str)
        spaces_needed = (24 - pot_size_length) // 2
        pot_size_display = " " * spaces_needed + pot_size_str + " " * (spaces_needed + pot_size_length % 2)

        round_pot_str1 = f"{self.round_pot[0]:.1f}bb"
        round_pot_length1 = len(round_pot_str1)
        spaces_needed1 = (20 - round_pot_length1) // 2
        round_pot_display1 = " " * spaces_needed1 + round_pot_str1 + " " * (spaces_needed1 + round_pot_length1 % 2)

        round_pot_str2 = f"{self.round_pot[1]:.1f}bb"
        round_pot_length2 = len(round_pot_str2)
        spaces_needed2 = (20 - round_pot_length2) // 2
        round_pot_display2 = " " * spaces_needed2 + round_pot_str2 + " " * (spaces_needed2 + round_pot_length2 % 2)
        
        info0 = "--> " if self.player_to_act == 0 else "" 
        if not P0_hide:
            info0 = info0 + "p0: " + " ".join([x.__repr__() for x in self.hands[0] if x is not None])
        else:
            info0 = info0 + "p0: " + " ".join(["??" for x in self.hands[0] if x is not None])

        info1 = "--> " if self.player_to_act == 1 else ""
        if not P1_hide:
            info1 = info1 + "p1: " + " ".join([x.__repr__() for x in self.hands[1] if x is not None])
        else:
            info1 = info1 + "p1: " + " ".join(["??" for x in self.hands[1] if x is not None])


        print()
        print(f"       {info0}")
        print(f"       {self.stacks[0]}bb  ")
        print("      ------------------")
        print(f"    /{round_pot_display1}\\" + str(" D" if self.button_position == 0 else ""))
        print(f"   /                      \\")
        print(f"  |{pot_size_display}|     ")
        print(f"   \\{community_cards_display}/")
        print(f"    \\{round_pot_display2}/" + str(" D" if self.button_position == 1 else ""))
        print("      ------------------")
        print(f"       {self.stacks[1]}bb  ")
        print(f"       {info1}")
        print()

