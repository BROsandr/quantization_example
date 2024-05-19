import unittest
from project.q_conv2d import q_conv2d
import torch
from torch.nn import functional as F
import torch.nn as nn
import random
import sys
import os
from project.QTensor import dequantize_tensor, quantize_tensor
from typing import Callable

# class TestBasic(unittest.TestCase):
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     # self.weight = torch.tensor([1

#   def test_const(self):
#     ...

class TestRandom(unittest.TestCase):
  SEED = random.randrange(sys.maxsize)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    random.seed(self.SEED)
    torch.manual_seed(self.SEED)
    self.MAX_RAND = 10
    self.ITER_NUM = 100

  def randomize(self):
    dim = random.randint(1, self.MAX_RAND)
    self.in_channels = random.randint(1, self.MAX_RAND)
    self.out_channels = random.randint(1, self.MAX_RAND)
    self.kernel_size = random.randint(1, dim)
    self.stride=random.randint(1, self.MAX_RAND)
    self.padding=random.randint(0, self.MAX_RAND)
    num_batches = random.randint(1,self.MAX_RAND )
    self.use_bias = bool(random.randint(0, 1))
    self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
    self.bias = torch.nn.Parameter(torch.randn(self.out_channels))
    self.input = torch.randn(num_batches, self.in_channels, dim, dim)

  def call(self, input, weight, bias):
    return F.conv2d(input=input, weight=weight, bias=(bias if self.use_bias else None), stride=self.stride, padding=self.padding)

  def run_iteration(self):
    self.randomize()
    actual: torch.Tensor=self.call(input=self.input, weight=self.weight, bias=self.bias)
    expected: torch.Tensor=dequantize_tensor(self.call(input=quantize_tensor(self.input), weight=quantize_tensor(self.weight), bias=quantize_tensor(self.bias)))
    self.assertTrue(torch.allclose(input=actual, other=expected, rtol=0.1, atol=0.1))

  def test_run(self):
    for _ in range(self.ITER_NUM): self.run_iteration()

if __name__ == '__main__':
  SEED = os.environ.get('TEST_SEED', TestRandom.SEED)
  TestRandom.SEED = int(SEED)

  unittest.main()
