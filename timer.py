import torch
import time

for _ in range(10):
    start = time.time()
    k = []
    a = torch.ze
    for _ in range(10000):
        a = torch.zeros(10)
        # a[:4] = 2
        # a[5] = 3
        # a[5:] = 4
        k.append(a)
    # a = torch.stack(k)
    print(time.time() - start)