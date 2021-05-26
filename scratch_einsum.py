import torch

x = torch.tensor(torch.arange(0,16)).view((4,4))
print(x)

y = torch.einsum('jj->', x)

print(y)

