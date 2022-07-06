# -*- coding: utf-8 -*-

import torch
import torch.optim as optim

loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
# loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
# loss_fn = torch.nn.MSELoss()
input = torch.autograd.Variable(torch.randn(3, 4))
target = torch.autograd.Variable(torch.randn(3, 4))
loss = loss_fn(input, target)
print(input);
print(target);
print(loss)
print(input.size(), target.size(), loss.size())