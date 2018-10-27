import torch
import numpy as np
import time

time_start = time.time()
a = np.eye(10000)
b = np.eye(10000)
for i in range(10):
    c = a.dot(b)

print(time.time() - time_start)
time_start = time.time()

torch.set_grad_enabled(False)
a_ = torch.from_numpy(a).to("cuda")
b_ = torch.from_numpy(b).to("cuda")
for i in range(10):
    c = torch.mm(a_, b_)

print(time.time() - time_start)