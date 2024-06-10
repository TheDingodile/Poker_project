from src.cardgames.NLHE import NLHE
from src.cardgames.Parallelized_NLHE import Parallelized_NLHE
from src.cardgames.PBS_NLHE import PBS_NLHE
from src.agents.random_agent import RandomAgent
from src.agents.keyboard_agent import KeyboardAgent
from src.agents.call_agent import CallAgent
from src.agents.fold_agent import FoldAgent
from src.agents.agent import Agent
import time
import matplotlib.pyplot as plt
from src.buffers.replay_buffer import ReplayBuffer


amount_values: int = 4 # min 1 max 13
amount_suits: int = 2 # min 1 max 4
cards_on_hand: int = 1
amount_community_cards: int = 5

stack_depth_bb: int = 100
refresh_stack: bool = True

reward_when_end_of_hand: bool = True
bet_sizes: list[float] = [0.2, 0.5, 1, 2] # bet sizes as a fraction of the pot

tables: int = 1
batch_size: int = 1

agents: list[Agent] = [CallAgent(bet_sizes), CallAgent(bet_sizes)]
replay_buffer: ReplayBuffer = ReplayBuffer(size=100000)
tables = [NLHE(amount_players=len(agents), stack_depth_bb=stack_depth_bb, amount_values=amount_values, amount_suits=amount_suits, cards_on_hand=cards_on_hand, amount_community_cards=amount_community_cards, refresh_stack=refresh_stack, reward_when_end_of_hand=reward_when_end_of_hand) for _ in range(tables)]
games = Parallelized_NLHE(amount_agents=len(agents), stack_depth_bb=stack_depth_bb, tables=tables)

# state, reward, done, info = games.new_hands() 
# for i in range(1000000):
#     print("played a total of", games.played_hands, "hands")
#     actions = games.take_actions(state, info, agents)
#     state, reward, done, info = games.step(actions)
#     # games.print_table()

PBS_games = PBS_NLHE(games, bet_sizes)
state, reward, done, info = PBS_games.new_hands() 

start = time.time()
for i in range(10):
    # time.sleep(1)

    PBS_games.print_table(P0_hide=False, P1_hide=False)
    print("played a total of", PBS_games.NLHE_games.played_hands, "hands")  
    actions = PBS_games.take_actions(state, agents)
    previous_state = state
    # print(actions)
    state, reward, dones, infos = PBS_games.step(actions)
    print(reward, dones)
    replay_buffer.add_data((previous_state, actions, reward, state, infos))
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
