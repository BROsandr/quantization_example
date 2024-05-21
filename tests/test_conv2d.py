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
from typing import Iterable, Sequence

import logging
logger = logging
logger.basicConfig(level=logging.DEBUG)

class Conv2dRandomizer:
  def __init__(self, SEED, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.SEED = SEED
    random.seed(self.SEED)
    torch.manual_seed(self.SEED)
    self.MAX_RAND = 10

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

class TestMyConv2d(unittest.TestCase):
  @staticmethod
  def my_conv2d(input, weight, bias = None, stride: int | Sequence[int] = 1, padding: int | Sequence[int] = 0):
    # see implementation in https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

    from math import floor

    if isinstance(stride, int): stride = [stride, stride]
    if isinstance(padding, int): padding = [padding, padding]
    dilation = [1, 1]

    inp_unf = torch.nn.functional.unfold(input=input, kernel_size=weight.shape[-2:], padding=padding, stride=stride)

    def mymatmul(input: torch.Tensor, other: torch.Tensor)->torch.Tensor:
      """
        Partial implementation of torch.matmul. Utilizes torch.mm.
      """

      is_valid_shape_len = lambda tensor: 2 <= len(tensor.shape) <= 4
      if not(is_valid_shape_len(input) and is_valid_shape_len(other)):
        raise ValueError(f'Unsupported shape. input.shape:{input.shape}, other.shape:{other.shape}')

      def normalize_shape(tensor: torch.Tensor)->int:
        num_of_extra_dims = 0
        while len(tensor.shape) != 4:
          tensor.unsqueeze_(0) # prepend dimension
          num_of_extra_dims += 1
        return num_of_extra_dims

      def denormalize_shape(tensor: torch.Tensor, num_of_extra_dims)->None:
        for _ in range(num_of_extra_dims): tensor.squeeze_(0)

      inp_num_of_extra_dims = normalize_shape(input)
      other_num_of_extra_dims = normalize_shape(other)

      from functools import partial

      denormalize_shape = partial(denormalize_shape, num_of_extra_dims=min(inp_num_of_extra_dims, other_num_of_extra_dims))

      are_dims_compatible = lambda dim1, dim2: (dim1 == dim2) or (dim1 == 1 or dim2 == 1)
      assert(all([are_dims_compatible(inp_dim, other_dim) for inp_dim, other_dim in zip(input.shape, other.shape)][0:2]))

      def get_broadcast_indices(dim1: int, dim2: int)->Iterable[tuple[int, int]]:
        from itertools import zip_longest
        return zip_longest(range(dim1), range(dim2), fillvalue=0) # for 1-dims an index will always be 0

      batch_out: list[list[torch.Tensor]] = []
      for inp_batch_idx, other_batch_idx in get_broadcast_indices(input.shape[0], other.shape[0]):
        channel_out: list[torch.Tensor] = []
        for inp_chan_idx, other_chan_idx in get_broadcast_indices(input.shape[1], other.shape[1]):
          channel_out.append(input[inp_batch_idx, inp_chan_idx, ...].mm(other[other_batch_idx, other_chan_idx, ...]))
        batch_out.append(channel_out)

      def cat_tensors(seq: Sequence)->torch.Tensor:
        if all(isinstance(el, torch.Tensor) for el in seq):
          return torch.stack(seq)
        return cat_tensors([cat_tensors(el) for el in seq])

      out = cat_tensors(batch_out)
      denormalize_shape(out)

      return out

    out_unf = mymatmul(inp_unf.transpose(1, 2), (weight.view(weight.size(0), -1).t())).transpose(1, 2)

    # see h_out, w_out formulas in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    h_out = floor((input.shape[-2] + 2 * padding[0] - dilation[0] * (weight.shape[-2] - 1) - 1) / stride[0] + 1)
    w_out = floor((input.shape[-1] + 2 * padding[1] - dilation[1] * (weight.shape[-1] - 1) - 1) / stride[1] + 1)
    out = torch.nn.functional.fold(input=out_unf, output_size=(h_out, w_out), kernel_size=(1, 1))
    if bias is not None: out += bias[..., None, None]
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

  def setUp(self):
    self.randomizer = Conv2dRandomizer(SEED=self.SEED)
    self.ITER_NUM = 100

  def call(self, input, weight, bias):
    randomizer = self.randomizer
    use_bias = randomizer.use_bias
    stride = randomizer.stride
    padding = randomizer.padding
    return F.conv2d(input=input, weight=weight, bias=(bias if use_bias else None), stride=stride, padding=padding)

  def run_iteration(self):
    randomizer = self.randomizer
    randomizer.randomize()
    input = randomizer.input
    weight = randomizer.weight
    bias = randomizer.bias
    actual: torch.Tensor=self.call(input=input, weight=weight, bias=bias)
    expected: torch.Tensor=dequantize_tensor(self.call(input=quantize_tensor(input), weight=quantize_tensor(weight), bias=quantize_tensor(bias)))
    self.assertTrue(torch.allclose(input=actual, other=expected, rtol=0.1, atol=0.1))

  def test_run(self):
    for i in range(self.ITER_NUM):
      with self.subTest(i=i): self.run_iteration()

if __name__ == '__main__':
  SEED = os.environ.get('TEST_SEED', TestRandom.SEED)
  TestRandom.SEED = int(SEED)

  unittest.main()
