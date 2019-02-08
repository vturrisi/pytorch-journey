import torch

c = torch.nn.Conv1d(2, 2, 2)

inp = torch.tensor(range(1, 5), dtype=torch.float).resize_(1, 2, 2)

max_value = 1 + sum(c.weight.size())
c.weight.data = torch.tensor(range(1, max_value), dtype=torch.float).resize_(*c.weight.size())
c.bias.data = torch.zeros(c.bias.size())

print(inp)
print(c.weight)
print(c(inp))
