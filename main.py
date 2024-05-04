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

stack_depth_bb: int = 100
tables = 10000
agents: list[Agent] = [RandomAgent(), RandomAgent()]
games = Parallelized_NLHE(amount_agents=len(agents), stack_depth_bb=stack_depth_bb, tables=tables)
state, reward, done, info = games.new_hands() 
# print(state, reward, done, info)
for i in range(1000000):
    # games.print_table(0)
    # print(games.get_action_space(0))
    print("played a total of", games.played_hands, "hands")
    actions = games.take_actions(state, info, agents)
    state, reward, done, info = games.step(actions)


# PBS_games = PBS_NLHE([NLHE(amount_players=len(agents), stack_depth_bb=stack_depth_bb) for _ in range(5)])
# for i in range(1000000):
#     state, reward, done, info = PBS_games.new_hand() 
#     PBS_games.print_table()
#     # print("starting game number", i)
#     while True:
#         action = agents[PBS_game.player_to_act].take_action_PBS(state, info)
#         state, reward, done, info = PBS_game.step(action)
#         PBS_game.print_table()
#         print(PBS_game.get_action_space())
#         if done:
#             break


























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
