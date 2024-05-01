import random

class Card():
    def __init__(self, suit: int, number: int):
        self.suit = suit
        self.number = number

    def __repr__(self):
        card_values = {11: "J", 12: "Q", 13: "K", 14: "A"}
        suit_values = {0: "♠", 1: "♥", 2: "♦", 3: "♣"}
        # suit_values = {0: "s", 1: "h", 2: "d", 3: "c"}
        number_repr = card_values.get(self.number, str(self.number))
        suit_repr = suit_values.get(self.suit)
        return f"{number_repr}{suit_repr}"
    
    def to_string(self):
        card_values = {11: "J", 12: "Q", 13: "K", 14: "A"}
        suit_values = {0: "s", 1: "h", 2: "d", 3: "c"}
        number_repr = card_values.get(self.number, str(self.number))
        suit_repr = suit_values.get(self.suit)
        return f"{number_repr}{suit_repr}" 
    
    def __eq__(self, other: 'Card'):
        return self.suit == other.suit and self.number == other.number
    

class Deck:
    def __init__(self):
        self.amount_suits = 4
        self.amount_values = 13
        self.reset()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()
    
    def burn_card(self):
        self.cards.pop()

    def reset(self):
        self.cards = [Card(suit, number) for suit in range(self.amount_suits) for number in range(2, 2 + self.amount_values)]
