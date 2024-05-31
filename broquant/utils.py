import torch.nn.functional as F
import torch
import torch.nn as nn
import functools
from typing import Callable, Iterable
from more_itertools import collapse
from dataclasses import dataclass

def train(args, model, device, train_loader, optimizer, epoch)->None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model: nn.Module, test_loader, device: torch.device | str ='cpu')->tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return (test_loss, accuracy)

class Implements:
  """Register a torch function override for Tensor"""

  def __init__(self, HANDLED_FUNCTIONS: dict):
    self.HANDLED_FUNCTIONS = HANDLED_FUNCTIONS

  def __call__(self, torch_function: Callable):
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        self.HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

def collapse_tensors(tensors: Iterable):
  return collapse(tensors, base_type=torch.Tensor)

@dataclass
class Metrics:
  abs_error   : float = 0.
  rel_error   : float = 0.
  max_expected: float = 0.

  @classmethod
  def eval(cls, expected: torch.Tensor, actual: torch.Tensor):
    max_expected = expected.max().item()
    abs_error = abs(expected - actual)
    rel_error = abs_error / (expected.abs() + torch.finfo(expected.dtype).tiny)
    max_abs_error = abs_error.max().item()
    max_rel_error = rel_error.max().item()
    return cls(
        abs_error=max_abs_error,
        rel_error=max_rel_error,
        max_expected=max_expected,
    )

  def __str__(self):
    return "\n".join((
      f'max_expected: {self.max_expected}',
      f'rel error   : {self.rel_error}',
      f'abs error   : {self.abs_error}',
    ))
