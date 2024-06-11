from src.cardgames.NLHE import NLHE
from src.cardgames.Parallelized_NLHE import Parallelized_NLHE
from src.cardgames.PBS_NLHE import PBS_NLHE
from src.agents.random_agent import RandomAgent
from src.agents.keyboard_agent import KeyboardAgent
from src.agents.call_agent import CallAgent
from src.agents.fold_agent import FoldAgent
from src.agents.raise_agent import RaiseAgent
from src.agents.random_nofold_agent import RandomNoFoldAgent
from src.agents.random_noraise_agent import RandomNoRaiseAgent
from src.agents.agent import Agent
import time
import matplotlib.pyplot as plt
from src.buffers.replay_buffer import ReplayBuffer
import torch


amount_values: int = 2 # min 1 max 13
amount_suits: int = 4 # min 1 max 4
cards_on_hand: int = 1
amount_community_cards: int = 6

stack_depth_bb: int = 100
refresh_stack: bool = True

reward_when_end_of_hand: bool = True
bet_sizes: list[float] = [100] # bet sizes as a fraction of the pot

tables: int = 1000
batch_size: int = 1

agents: list[Agent] = [RandomNoFoldAgent(bet_sizes), RandomNoFoldAgent(bet_sizes)]
replay_buffer: ReplayBuffer = ReplayBuffer(size=100000)
tables = [NLHE(amount_players=len(agents), stack_depth_bb=stack_depth_bb, amount_values=amount_values, amount_suits=amount_suits, cards_on_hand=cards_on_hand, amount_community_cards=amount_community_cards, refresh_stack=refresh_stack, reward_when_end_of_hand=reward_when_end_of_hand) for _ in range(tables)]
games = Parallelized_NLHE(amount_agents=len(agents), stack_depth_bb=stack_depth_bb, tables=tables)

# state, reward, done, info = games.new_hands() 
# for i in range(20):
#     print("played a total of", games.played_hands, "hands")
#     games.print_table(table_number=0)
#     actions = games.take_actions(state, info, agents)
#     state, reward, done, info = games.step(actions)
#     print(reward, done)
#     time.sleep(0.5)

PBS_games = PBS_NLHE(games, bet_sizes)
state, reward, done, info = PBS_games.new_hands() 

start = time.time()
total_reward = [0, 0]
for i in range(1000):
    # time.sleep(1)

    PBS_games.print_table(P0_hide=False, P1_hide=False)
    print("played a total of", PBS_games.NLHE_games.played_hands, "hands")  
    actions = PBS_games.take_actions(state, agents)
    previous_state = state
    state, reward, dones, infos = PBS_games.step(actions)
    print(reward[0], dones[0])
    replay_buffer.add_data((previous_state, actions, reward, state, infos))
    total_reward = [total_reward[0] + torch.sum(reward[:, :, 0]), total_reward[1] + torch.sum(reward[:, :, 1])]
    blinds_fee = sum(dones) * 0.75
    total_reward = [total_reward[0] - blinds_fee, total_reward[1] - blinds_fee]
    print(total_reward, torch.sum(torch.tensor(total_reward)))
    # quit()

print(time.time() - start)


























# game = NLHE(amount_players=len(agents), stack_depth_bb=stack_depth_bb)  
# for i in range(1000000):
#     state, reward, done, info = game.new_hand() 
#     # game.print_table()
#     # print(game.get_action_space())
#     if i % 10000 == 0:
#         print("starting hand number", i)
#     while True:
#         action = agents[game.player_to_act].take_action(state, info)
#         state, reward, done, info = game.step(action)
#         # game.print_table()
#         # print(game.get_action_space())
#         if done:
#             break
