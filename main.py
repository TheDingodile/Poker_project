from src.cardgames.NLHE import NLHE
from src.agents.random_agent import RandomAgent
from src.agents.keyboard_agent import KeyboardAgent
from src.agents.agent import Agent

stack_depth_bb = 100
agents: list[Agent] = [KeyboardAgent(), KeyboardAgent()]

game = NLHE(amount_players=len(agents), stack_depth_bb=stack_depth_bb)


for _ in range(100):
    state, reward, done, info = game.new_hand() 
    while True:
        print(state)
        action = agents[game.player_to_act].take_action(state)
        state, reward, done, info = game.step(action)
        if done:
            break
    print(f"done {done}")
    print(f"reward {reward}")
    # print(info)
    # print(game.stacks)
    


