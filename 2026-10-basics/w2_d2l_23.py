import torch

A=torch.arange(6,dtype=torch.float32).reshape(2,3)

print(torch.norm(A))



