import random
import torch
from torch import Tensor
import time

class ReplayBuffer():
    def __init__(self, size: int = 10000) -> None:
        self.size = size
        self.buffer = [None for _ in range(size)]
        self.index = 0
        self.current_size = -1



    def add_data(self, data: tuple[Tensor], info: list[dict]) -> None:
        if self.current_size == -1:
            if info[0]["player_to_act"] == 1:
                return
            self.current_states = data[0]
            self.current_actions = data[1]
            self.current_rewards = data[2][:, :, 0]
            self.current_dones = data[3]
            self.current_size = 0
        else:
            idx_of_acting_player = [info[i]["player_to_act"] for i in range(len(info))]
            # print(idx_of_acting_player)
            indexes = torch.where(torch.tensor(idx_of_acting_player) == 0)
            print(self.current_rewards[0])
            self.current_rewards[~self.current_dones] += data[2][~self.current_dones, :, 0]
            print(self.current_rewards[0])
            self.current_dones = torch.logical_or(self.current_dones, data[3])
            # print(torch.sum(self.current_dones), torch.sum(data[3]))

            for idx in indexes[0]:
                # print(self.current_states.shape, self.current_states[idx].shape, self.current_actions[idx].shape, data[0][idx].shape, self.current_rewards[idx].shape, self.current_dones[idx])
                # print(self.current_dones[idx])

                d = (self.current_states[idx].detach().clone(), self.current_actions[idx].detach().clone(), data[0][idx].detach().clone(), self.current_rewards[idx].detach().clone(), self.current_dones[idx].detach().clone())
                self.buffer[self.index] = d
                self.current_states[idx] = data[0][idx]
                self.current_actions[idx] = data[1][idx]
                self.current_rewards[idx] = data[2][idx, :, 0] * (~self.current_dones[idx])
                # print(self.buffer[self.index][4])
                self.current_dones[idx] = 0
                self.index = (self.index + 1) % self.size
                self.current_size = min(self.current_size + 1, self.size)

            # print(torch.sum(self.current_dones))


    def sample(self, batch_size: int) -> list[tuple]:

        batch_size = min(batch_size, self.current_size)
        indices = random.sample(range(self.current_size), batch_size)

        return [torch.stack([self.buffer[i][j] for i in indices]) for j in range(len(self.buffer[0]))]
