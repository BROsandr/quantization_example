import torch.nn.functional as F
import torch
import torch.nn as nn
import functools
from typing import Callable, Iterable, Any
from collections.abc import Mapping
from more_itertools import collapse
from enum import Enum

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

class Metrics(Enum):
  ABS_ERROR    = "absolute error"
  REL_ERROR    = "relative error"
  MAX_EXPECTED = "max expected"
  ATOL         = "absolute tolerance"

def eval_metrics(expected: torch.Tensor, actual: torch.Tensor)->dict:
  metrics = {}
  metrics[Metrics.MAX_EXPECTED] = expected.max().item()
  abs_error = abs(expected - actual)
  rel_error = abs_error / (expected.abs() + torch.finfo(expected.dtype).tiny)
  metrics[Metrics.ABS_ERROR] = abs_error.max().item()
  metrics[Metrics.REL_ERROR] = rel_error.max().item()
  return metrics

def metrics2str(metrics: Mapping[Metrics, Any]):
  return "\n".join(f'{metric.value}: {val}' for metric, val in metrics.items())
