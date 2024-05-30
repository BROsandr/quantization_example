import unittest
import sys
import torch
from torch.nn import functional as F
import torch.nn as nn
import random
import os
from dataclasses import dataclass

if __name__ == '__main__': sys.path.append('.')
from broquant.QTensor import QTensor
from broquant.TolTensor import TolTensor

import logging
logger = logging
logger.basicConfig(level=logging.DEBUG)

_SEED = random.randrange(sys.maxsize)
def set_default_seed():
  global _SEED
  seed = os.environ.get('TEST_SEED', _SEED)
  _SEED = int(seed)
  random.seed(_SEED)
  torch.manual_seed(_SEED)
  logger.debug(f'SEED:{_SEED}')

set_default_seed()

@dataclass
class Stats:
  abs_error: float = 0.
  rel_error: float = 0.

def print_stats(expected: torch.Tensor, actual: torch.Tensor, atol: float)->Stats:
  logger.debug(f"atol:{atol}")
  max_expected = expected.max().item()
  abs_error = abs(expected - actual)
  rel_error = abs_error / expected.abs()
  max_abs_error = abs_error.max().item()
  max_rel_error = rel_error.max().item()
  logger.debug(f"max expected:{max_expected}")
  logger.debug(f"max rel error:{max_rel_error}")
  logger.debug(f"max abs error:{max_abs_error}")

  return Stats(abs_error=max_abs_error, rel_error=max_rel_error)

class LinearRandomizer:
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.MAX_RAND = 10

  def randomize(self):
    dim_w = random.randint(1, self.MAX_RAND)
    dim_h = random.randint(1, self.MAX_RAND)
    self.in_channels = random.randint(1, self.MAX_RAND)
    self.out_channels = random.randint(1, self.MAX_RAND)
    num_batches = random.randint(1,self.MAX_RAND )
    self.use_bias = bool(random.randint(0, 1))
    self.weight = torch.nn.Parameter(torch.rand(self.out_channels, dim_w, requires_grad=False))
    self.bias = torch.nn.Parameter(torch.rand(self.out_channels, requires_grad=False))
    self.input = torch.rand(num_batches, self.in_channels, dim_h, dim_w, requires_grad=False)

def calc_max_linear_atol(input: QTensor, weight: QTensor, bias=None, conv2d=F.linear)->float:
  input_tol_tensor = TolTensor.from_QTensor(input)
  weight_tol_tensor = TolTensor.from_QTensor(weight)
  bias_tol_tensor = TolTensor.from_QTensor(bias) if not (bias is None) else None
  return conv2d(input=input_tol_tensor, weight=weight_tol_tensor, bias=bias_tol_tensor).atol

class TestConst(unittest.TestCase):
  def setUp(self):
    self.x = torch.tensor(
        [[[[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [9, 10, 11, 12],
        [13, 14, 15, 16]],
        [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [9, 10, 11, 12],
        [13, 14, 15, 16]]],
        [[[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [9, 10, 11, 12],
        [13, 14, 15, 16]],
        [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [9, 10, 11, 12],
        [13, 14, 15, 16]]]],
        dtype=torch.float32
    )
    self.weight = torch.tensor(
       [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
        [0, -1, 0]],
        dtype=torch.float32
    ).t()

    self.bias = torch.tensor([5., 4., 6.])

  def test_weight_only(self):
    with torch.no_grad():
      expected = F.linear(self.x, weight=self.weight)
      q_input = QTensor.quantize(self.x)
      q_weight = QTensor.quantize(self.weight)
      actual = F.linear(q_input, q_weight).dequantize()

    self.assertTrue(torch.allclose(input=actual, other=expected, atol=calc_max_linear_atol(input=q_input, weight=q_weight)))

  def test_bias(self):
    with torch.no_grad():
      expected = F.linear(self.x, weight=self.weight, bias=self.bias)
      q_input = QTensor.quantize(self.x)
      q_weight = QTensor.quantize(self.weight)
      q_bias = QTensor.quantize(self.bias, scale=(q_input.scale*q_weight.scale), zero_point=0, dtype=torch.int32)
      actual = F.linear(q_input, q_weight, bias=q_bias).dequantize()

    self.assertTrue(torch.allclose(input=actual, other=expected, atol=calc_max_linear_atol(input=q_input, weight=q_weight, bias=q_bias)))

class TestRandom(unittest.TestCase):
  def setUp(self):
    self.randomizer = LinearRandomizer()
    self.ITER_NUM = 100
    self.max_stats = Stats()

  def call(self, input, weight, bias):
    randomizer = self.randomizer
    use_bias = randomizer.use_bias
    return F.linear(input=input, weight=weight, bias=(bias if use_bias else None))

  def run_iteration(self):
    randomizer = self.randomizer
    randomizer.randomize()
    input = randomizer.input
    weight = randomizer.weight
    bias = randomizer.bias
    with torch.no_grad():
      expected: torch.Tensor=self.call(input=input, weight=weight, bias=bias)
      q_input = QTensor.quantize(input)
      q_weight = QTensor.quantize(weight)
      q_bias = QTensor.quantize(bias, scale=(q_input.scale*q_weight.scale), zero_point=0, dtype=torch.int32)
      actual: torch.Tensor=self.call(input=q_input, weight=q_weight, bias=q_bias).dequantize()
    atol=calc_max_linear_atol(input=q_input, weight=q_weight, bias=q_bias, conv2d=self.call)
    stats = print_stats(actual=actual, atol=atol, expected=expected)
    self.max_stats.abs_error, self.max_stats.rel_error = max(self.max_stats.abs_error, stats.abs_error), max(self.max_stats.rel_error, stats.rel_error)
    self.assertGreaterEqual(expected.abs().max().item(), atol) # sanity check
    self.assertTrue(torch.allclose(input=actual, other=expected, atol=atol))

  def test_run(self):
    for i in range(self.ITER_NUM):
      with self.subTest(i=i): self.run_iteration()
    logger.info(f'For all iteration. max abs error:{self.max_stats.abs_error}, max rel error:{self.max_stats.rel_error}')

if __name__ == '__main__':
  unittest.main()
