from src.cardgames.NLHE import NLHE
from src.agents.random_agent import RandomAgent
from src.agents.keyboard_agent import KeyboardAgent
from src.agents.call_agent import CallAgent
from src.agents.fold_agent import FoldAgent
from src.agents.agent import Agent
import time
import matplotlib.pyplot as plt

stack_depth_bb: int = 100
agents: list[Agent] = [RandomAgent(), RandomAgent()]
game = NLHE(amount_players=len(agents), stack_depth_bb=stack_depth_bb)  

for i in range(1000000):
    state, reward, done, info = game.new_hand() 
    # print("starting hand number", i)
    while True:
        action = agents[game.player_to_act].take_action(state, info)
        # game.print_table()
        state, reward, done, info = game.step(action)
        if done:
            break

