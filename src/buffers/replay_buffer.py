import random
import torch
from torch import Tensor

class ReplayBuffer():
    def __init__(self, size: int = 10000) -> None:
        self.size = size
        self.buffer = [None for _ in range(size)]
        self.index = 0
        self.current_size = 0

    def add_data(self, data: tuple[Tensor]) -> None:
        for i in range(len(data[0])):
            self.buffer[self.index] = tuple([data[j][i].detach() for j in range(len(data))])
            self.index = (self.index + 1) % self.size
            self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size: int) -> list[tuple]:
        batch_size = min(batch_size, self.current_size)
        indices = random.sample(range(self.current_size), batch_size)
        # pick elements and stack them using torch.stack for each element in the tuple
        return [torch.stack([self.buffer[i][j] for i in indices]) for j in range(len(self.buffer[0]))]
