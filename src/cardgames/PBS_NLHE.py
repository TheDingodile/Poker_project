from src.cardgames.NLHE import NLHE
from src.cardgames.StandardDeck import Card, Deck
import torch

class PBS_NLHE:
    def __init__(self, NLHE_games: list[NLHE]) -> None:
        self.NLHE_games = NLHE_games
        self.amount_cards = self.NLHE_games[0].deck.amount_suits * self.NLHE_games[0].deck.amount_values
        self.infostates: int = self.amount_cards * (self.amount_cards - 1) // 2
        self.hand_idx_to_infostate_idx_map = self.create_infostate_map()
        self.infostate_idx_to_hand_idx_map = {v: k for k, v in self.hand_idx_to_infostate_idx_map.items()}

    def get_hand_idx(self, hand: list[Card]) -> int:
        return hand[0].id * self.amount_cards + hand[1].id

    def create_infostate_map(self) -> dict[int, int]:
        hand_to_infostate_map = {}
        for i in range(self.amount_cards):
            for j in range(i + 1, self.amount_cards):
                hand_to_infostate_map[i * self.amount_cards + j] = len(hand_to_infostate_map)

        for i in range(self.amount_cards):
            for j in range(i):
                hand_to_infostate_map[i * self.amount_cards + j] = hand_to_infostate_map[j * self.amount_cards + i]

        return hand_to_infostate_map

    def hand_to_infostate_idx(self, hand: list[Card]) -> int:
        return self.hand_idx_to_infostate_idx_map[self.get_hand_idx(hand)]
    
    def infostate_idx_to_hand(self, idx: int) -> list[Card]:
        hand_id = self.infostate_idx_to_hand_idx_map[idx]
        card1_id = hand_id // self.amount_cards
        card2_id = hand_id % self.amount_cards
        return [Card(card1_id % self.NLHE_games[0].deck.amount_suits, card1_id // self.NLHE_games[0].deck.amount_suits + 2), Card(card2_id % self.NLHE_game.deck.amount_suits, card2_id // self.NLHE_game.deck.amount_suits + 2)]


    def new_hand(self):
        state, reward, done, info = self.NLHE_games.new_hand()
        state["PBS"] = torch.ones(size=(self.infostates, self.NLHE_games.amount_players)) / self.infostates
        return state, torch.zeros(size=(self.infostates, self.NLHE_games.amount_players)), torch.zeros(self.infostates), info
    
    def print_table(self, P0_hide=True, P1_hide=True):
        self.NLHE_games[0].print_table(P0_hide=P0_hide, P1_hide=P1_hide)