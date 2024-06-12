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
from src.agents.true_random_agent import TrueRandomAgent
from src.agents.NN_agent import NNAgent
from src.agents.agent import Agent
import time
import matplotlib.pyplot as plt
from src.buffers.replay_buffer import ReplayBuffer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


amount_values: int = 4 # min 1 max 13
amount_suits: int = 2 # min 1 max 4
cards_on_hand: int = 1
amount_community_cards: int = 5

stack_depth_bb: int = 100
refresh_stack: bool = True

reward_when_end_of_hand: bool = True
bet_sizes: list[float] = [1] # bet sizes as a fraction of the pot

tables: int = 1000
batch_size: int = 100

lr: float = 0.001
update_target_every: int = 100
plot_every: int = 100

agents: list[Agent] = [NNAgent(bet_sizes, lr=lr, update_target_every=update_target_every), RandomNoFoldAgent(bet_sizes)]
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

for i in range(1000):
    # PBS_games.print_table(P0_hide=False, P1_hide=False)
    print("played a total of", PBS_games.NLHE_games.played_hands, "hands")  
    actions = PBS_games.take_actions(state, agents)
    previous_state = state
    state, reward, dones, infos = PBS_games.step(actions)
    # print(infos)
    replay_buffer.add_data((previous_state, actions, state, reward, dones), infos)

    replay_buffer_sample = replay_buffer.sample(batch_size)
    agents[0].train_DDPG(replay_buffer_sample, i)

    if (i + 1) % plot_every == 0:
        print("plotting")
        # make plot that close itself after 5 seconds
        data = PBS_games.reward_list
        # do running average of the last 100 games
        data = [sum(data[i:i+10])/10 for i in range(len(data)-10)]
        plt.plot(data)
        plt.show(block=False)
        plt.pause(5)
        plt.close()

print(time.time() - start)





















