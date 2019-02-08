import torch

w = torch.tensor([1, 1, 0, 1e-200000], requires_grad=True)
o = torch.sum(torch.log(w + 1e-14))
loss = 10 - o
print(o, loss)
loss.backward()
print(o.grad)
print(w.grad)
