from phevaluator.evaluator import evaluate_cards
from src.cardgames.StandardDeck import Card, Deck
import time
import torch


# batch_size = 2
# infostates = 4
# players = 2

# hand_to_impossible_infostates = {i: [i] for i in range(infostates)}
# infostate_to_possible_infostates = torch.ones(batch_size, infostates, infostates)
# for hand in hand_to_impossible_infostates:
#     for impossible_infostate in hand_to_impossible_infostates[hand]:
#         infostate_to_possible_infostates[:, hand, impossible_infostate] = 0
# print("started")
# start_time = time.time()

# avg_equity = 0
# for j in range(100):
#     # make a random PBS
#     PBS = torch.rand(batch_size, infostates, players)
#     PBS /= PBS.sum(dim=1, keepdim=True)

#     # random permutation of length infostates

#     permutation = torch.stack([torch.randperm(infostates) for _ in range(batch_size)], dim=0)
#     # stack permutation to batch size
#     # permutation = torch.cat((permutation.unsqueeze(0), torch.arange(infostates).unsqueeze(0)), dim=0)
#     print(permutation)
#     print(PBS)
#     # make identity permutation
#     # permutation = torch.arange(infostates)

#     permuted_PKS = PBS[:, permutation, :]
#     # infostate_to_possible_infostates_permuted = infostate_to_possible_infostates[:, :, permutation]

#     permuted_PKS_p1 = permuted_PKS[:, :, 0]
#     permuted_PKS_p2 = permuted_PKS[:, :, 1]

#     # multiply the PBS with the infostate_to_possible_infostates to get the infostate matrix, unsqueeze to make it work
#     infostate_matrix = PBS[:,:,1:2].transpose(2, 1) * infostate_to_possible_infostates
#     expanded_permutation = permutation.unsqueeze(1).expand(infostate_matrix.shape)

#     infostate_matrix_permuted = torch.gather(infostate_matrix, 2, expanded_permutation)
#     cum_sum_info_matrix = torch.cumsum(infostate_matrix_permuted, dim=2)
#     argsorted_permutation = torch.argsort(permutation)

#     equity_per_infostate_unnormalized = torch.gather(cum_sum_info_matrix, 2, argsorted_permutation.unsqueeze(2)).squeeze(2)
#     equity_per_infostate = equity_per_infostate_unnormalized / cum_sum_info_matrix[:, :, -1]
#     print(equity_per_infostate)
#     equity_p0 = torch.sum(equity_per_infostate * PBS[:, :, 0], dim=1)
#     avg_equity += equity_p0.mean().item()
#     quit()

# print(avg_equity / 100)
# print("--- %s seconds ---" % (time.time() - start_time))





deck = Deck(13, 4)
hand = [deck.deal_card(), deck.deal_card(), deck.deal_card(), deck.deal_card(), deck.deal_card(), deck.deal_card()]
t = [card.to_string() for card in hand]

for j in range(1100):
    for i in range(1000):
        print(evaluate_cards('2c', '2h', '2h', '2h', '2s', '2s', '2h'))
