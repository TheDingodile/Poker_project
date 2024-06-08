from phevaluator.evaluator import evaluate_cards
from src.cardgames.StandardDeck import Card, Deck
import time


start = time.time()

deck = Deck(13, 4)

hand = [deck.deal_card(), deck.deal_card(), deck.deal_card(), deck.deal_card(), deck.deal_card(), deck.deal_card()]
t = [card.to_string() for card in hand]
print(t)
for i in range(100000):
    # j = i
    evaluate_cards("Ac", "Ad", "Ah", "As", "Kc", "Qh")

print(time.time() - start)