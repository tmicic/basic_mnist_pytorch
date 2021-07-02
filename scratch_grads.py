import torch

x = torch.tensor(3., requires_grad=True)
y = torch.tensor(4., requires_grad=True)

z = 3 * x ** 3 + y


print(z)

z.backward()

print(x.grad)
print(y.grad)

x = x - ((1) * x.grad)

print(x)
