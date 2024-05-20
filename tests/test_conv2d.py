import unittest
import sys
if __name__ == '__main__': sys.path.append('.')
from broquant.q_conv2d import q_conv2d
import torch
from torch.nn import functional as F
import torch.nn as nn
import random
import os
from broquant.QTensor import dequantize_tensor, quantize_tensor
from typing import Sequence

import logging
logger = logging
logger.basicConfig(level=logging.DEBUG)

class TestMyConv2d(unittest.TestCase):
  @staticmethod
  def my_conv2d(input, weight, bias = None, stride: int | Sequence[int] = 1, padding: int | Sequence[int] = 0):
    # see implementation in https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

    from math import floor

    if isinstance(stride, int): stride = [stride, stride]
    if isinstance(padding, int): padding = [padding, padding]
    dilation = [1, 1]

    inp_unf = torch.nn.functional.unfold(input=input, kernel_size=weight.shape[-2:], padding=padding, stride=stride)
    out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)

    # see h_out, w_out formulas in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    h_out = floor((input.shape[-2] + 2 * padding[0] - dilation[0] * (weight.shape[-2] - 1) - 1) / stride[0] + 1)
    w_out = floor((input.shape[-1] + 2 * padding[1] - dilation[1] * (weight.shape[-1] - 1) - 1) / stride[1] + 1)
    out = torch.nn.functional.fold(input=out_unf, output_size=(h_out, w_out), kernel_size=(1, 1))
    if bias is not None:
      bias_tensor = torch.ones(out.shape)
      for channel in range(bias_tensor.shape[-3]):
        bias_tensor[..., channel, :, :] *= bias[channel]
      out += bias_tensor
    return out

  def setUp(self):
    self.x = torch.tensor(
        [[[[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]],
        [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]]],
        [[[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]],
        [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]]]],
        dtype=torch.float32
    )
    self.weight = torch.tensor(
        [[[[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
        [0, -1, 0]],
        [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
        [0, -1, 0]]],
        [[[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
        [0, -1, 0]],
        [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
        [0, -1, 0]]]],
        dtype=torch.float32
    )

    self.bias = torch.tensor([5., 4.])

  def test_weight_only(self):

    expected = F.conv2d(self.x, weight=self.weight)

    actual = self.my_conv2d(self.x, self.weight)

    self.assertTrue(torch.all(torch.eq(expected, actual)))

  def test_bias(self):
    expected = F.conv2d(self.x, weight=self.weight, bias=self.bias)

    actual = self.my_conv2d(self.x, self.weight, bias=self.bias)

    self.assertTrue(torch.all(torch.eq(expected, actual)))

  def test_equal_padding(self):
    padding = 1
    expected = F.conv2d(self.x, weight=self.weight, bias=self.bias, padding=padding)
    actual = self.my_conv2d(self.x, self.weight, bias=self.bias, padding=padding)
    self.assertTrue(torch.all(torch.eq(expected, actual)))

  def test_unequal_padding(self):
    padding = [1, 2]
    expected = F.conv2d(self.x, weight=self.weight, bias=self.bias, padding=padding)
    actual = self.my_conv2d(self.x, self.weight, bias=self.bias, padding=padding)
    self.assertTrue(torch.all(torch.eq(expected, actual)))

  def test_equal_stride(self):
    stride = 2
    padding = 1
    expected = F.conv2d(self.x, weight=self.weight, bias=self.bias, padding=padding, stride=stride)
    actual = self.my_conv2d(self.x, self.weight, bias=self.bias, padding=padding, stride=stride)
    self.assertTrue(torch.all(torch.eq(expected, actual)))

  def test_unequal_stride(self):
    stride = [2, 3]
    padding = 1
    expected = F.conv2d(self.x, weight=self.weight, bias=self.bias, padding=padding, stride=stride)
    actual = self.my_conv2d(self.x, self.weight, bias=self.bias, padding=padding, stride=stride)
    self.assertTrue(torch.all(torch.eq(expected, actual)))

class TestConst(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def test_single_stride_no_bias(self):
    x = torch.tensor(
      [[[1, 2],
      [3, 4]]],
      dtype=torch.float32
    )
    weight = torch.tensor(
      [[[[1, 1],
      [2, 2]]]],
      dtype=torch.float32
    )
    bias = torch.tensor([0.])

    y = F.conv2d(x, weight=weight, bias=bias)
    self.assertTrue(torch.all(torch.eq(y, torch.tensor([[[17.]]]))))

    q_y = F.conv2d(quantize_tensor(x), weight=quantize_tensor(weight), bias=quantize_tensor(bias))
    self.assertTrue(torch.all(torch.eq(dequantize_tensor(q_y), torch.tensor([[[17.]]]))))

  def test_const(self):
    ...

@unittest.expectedFailure
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
    for i in range(self.ITER_NUM):
      with self.subTest(i=i): self.run_iteration()

if __name__ == '__main__':
  SEED = os.environ.get('TEST_SEED', TestRandom.SEED)
  TestRandom.SEED = int(SEED)

  unittest.main()
