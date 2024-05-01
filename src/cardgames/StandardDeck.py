import random

class Card:
    def __init__(self, suit: int, number: int):
        self.suit = suit
        self.number = number

    def __repr__(self):
        card_values = {11: "J", 12: "Q", 13: "K", 14: "A"}
        suit_values = {0: "♠", 1: "♥", 2: "♦", 3: "♣"}
        number_repr = card_values.get(self.number, str(self.number))
        suit_repr = suit_values.get(self.suit)
        return f"{number_repr}{suit_repr}"

class Deck:
    def __init__(self):
        self.reset()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()
    
    def burn_card(self):
        self.cards.pop()

    def reset(self):
        self.cards = [Card(suit, number) for suit in range(4) for number in range(2, 15)]
