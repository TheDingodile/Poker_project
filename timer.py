import torch
import time


a = torch.ones(size=(10000, 10000, 2))
b = torch.arange(10000)
c = torch.randint(2, b.shape)

mask = c == 0
mask2 = c == 1

for _ in range(10):
    start = time.time()
    for _ in range(10):
        d = a[torch.arange(10000), :, c]
        # d = a[mask, :, 0]
        # d = a[mask2, :, 1]
    
    print(time.time() - start)