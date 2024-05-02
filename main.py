from src.cardgames.NLHE import NLHE
from src.agents.random_agent import RandomAgent
from src.agents.keyboard_agent import KeyboardAgent
from src.agents.call_agent import CallAgent
from src.agents.agent import Agent
import time

stack_depth_bb = 100
bet_sizes = [0.2, 0.5, 1, 2]
agents: list[Agent] = [RandomAgent(bet_sizes=bet_sizes), RandomAgent(bet_sizes=bet_sizes)]

game = NLHE(amount_players=len(agents), stack_depth_bb=stack_depth_bb)

for _ in range(100):
    state, reward, done, info = game.new_hand() 
    while True:
        time.sleep(1)
        action = agents[game.player_to_act].take_action(state, info)
        print(action)
        state, reward, done, info = game.step(action)
        if done:
            break
    # print(f"rewards {reward}")
    # print(info)
    # print(game.stacks)
# measure time

    


