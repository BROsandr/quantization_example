import unittest
import sys
import torch
from torch.nn import functional as F
import torch.nn as nn
import random
import os

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

if __name__ == '__main__':
  unittest.main()
