import torch

x = torch.tensor(32., requires_grad=True)
y = torch.tensor(3., requires_grad=False)

z = x * y


# we want y = 0 therefore loss = y

z.backward()

print(x.grad)

#print(x.grad)
#print(y.grad)
#print(z.grad)

