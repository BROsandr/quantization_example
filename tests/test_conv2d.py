import unittest
import sys
import torch
from torch.nn import functional as F
import torch.nn as nn
import random
import os
from broquant.QTensor import dequantize_tensor, quantize_tensor, QTensor
from broquant.TolTensor import TolTensor
from broquant.utils import eval_metrics, Metrics, metrics2str
from typing import Iterable, Any

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

_SEED = random.randrange(sys.maxsize)
def set_default_seed():
  global _SEED
  seed = os.environ.get('TEST_SEED', _SEED)
  _SEED = int(seed)
  random.seed(_SEED)
  torch.manual_seed(_SEED)
  logger.debug(f'SEED:{_SEED}')

set_default_seed()

def calc_max_mm_atol(a: QTensor, b: QTensor)->float:
  a_tol_tensor = TolTensor.from_QTensor(a)
  b_tol_tensor = TolTensor.from_QTensor(b)
  return a_tol_tensor.mm(b_tol_tensor).atol

def calc_max_conv2d_atol(input: QTensor, weight: QTensor, bias=None, conv2d=F.conv2d)->float:
  input_tol_tensor = TolTensor.from_QTensor(input)
  weight_tol_tensor = TolTensor.from_QTensor(weight)
  bias_tol_tensor = TolTensor.from_QTensor(bias) if not (bias is None) else None
  return conv2d(input=input_tol_tensor, weight=weight_tol_tensor, bias=bias_tol_tensor).atol

class Conv2dRandomizer:
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
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
    self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=False))
    self.bias = torch.nn.Parameter(torch.randn(self.out_channels, requires_grad=False))
    self.input = torch.randn(num_batches, self.in_channels, dim, dim, requires_grad=False)

class TestMyConv2d(unittest.TestCase):
  @staticmethod
  def my_conv2d(input, weight, bias = None, stride: Any = 1, padding: Any = 0)->torch.Tensor:
    # see implementation in https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

    from math import floor

    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
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

      def cat_tensors(seq: tuple | list)->torch.Tensor:
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
    self.ITER_NUM = 100
    self.randomizer = Conv2dRandomizer()

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

  def test_random(self):
    for i in range(self.ITER_NUM):
      with self.subTest(i=i): self.rand_iter()

  def rand_iter(self):
    randomizer = self.randomizer
    randomizer.randomize()

    input = randomizer.input
    stride = randomizer.stride
    padding = randomizer.padding
    bias = randomizer.bias if randomizer.use_bias else None
    weight = randomizer.weight

    with torch.no_grad():
      expected = F.conv2d(input, weight=weight, bias=bias, padding=padding, stride=stride)
      actual = self.my_conv2d(input, weight=weight, bias=bias, padding=padding, stride=stride)
    cmp_res = torch.allclose(other=expected, input=actual, atol=1e-04) # it is unexpected that there is atol. Probably due to python-c differences.
    self.assertTrue(cmp_res)

class TestMyMM(unittest.TestCase):
  @staticmethod
  def mymm(input: torch.Tensor, mat2: torch.Tensor):
    ar,ac = input.shape
    br,bc = mat2.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            c[i,j] = (input[i,:] * mat2[:,j]).sum() # multiply all of column j by all of row i and sum it
    return c

  def test_2x2(self):
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8, requires_grad=False)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.uint8, requires_grad=False)
    with torch.no_grad():
      expected = torch.mm(a, b)
      actual = self.mymm(a, b)
    cmp_res = torch.all(torch.eq(expected, actual))
    self.assertTrue(cmp_res)

class TestMyMMQuant(unittest.TestCase):
  def test_2x2(self):
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8, requires_grad=False)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.uint8, requires_grad=False)
    with torch.no_grad():
      expected = torch.mm(a, b)
      actual = quantize_tensor(a).mm(quantize_tensor(b))
    cmp_res = torch.allclose(other=expected.to(torch.float32), input=dequantize_tensor(actual))
    self.assertTrue(cmp_res)

  def test_const1(self):
    a = torch.tensor([[ 1.,  2.,  3.,  5.,  6.,  7.,  9., 10., 11.,  9., 10., 11.,  1.,  2.,
          3.,  5.,  6.,  7.,  9., 10., 11.,  9., 10., 11.],
        [ 2.,  3.,  4.,  6.,  7.,  8., 10., 11., 12., 10., 11., 12.,  2.,  3.,
          4.,  6.,  7.,  8., 10., 11., 12., 10., 11., 12.],
        [ 5.,  6.,  7.,  9., 10., 11.,  9., 10., 11., 13., 14., 15.,  5.,  6.,
          7.,  9., 10., 11.,  9., 10., 11., 13., 14., 15.],
        [ 6.,  7.,  8., 10., 11., 12., 10., 11., 12., 14., 15., 16.,  6.,  7.,
          8., 10., 11., 12., 10., 11., 12., 14., 15., 16.]])
    b = torch.tensor(
       [[ 0.,  0.],
        [-1., -1.],
        [ 0.,  0.],
        [-1., -1.],
        [ 5.,  5.],
        [-1., -1.],
        [ 0.,  0.],
        [-1., -1.],
        [ 0.,  0.],
        [ 0.,  0.],
        [-1., -1.],
        [ 0.,  0.],
        [ 0.,  0.],
        [-1., -1.],
        [ 0.,  0.],
        [-1., -1.],
        [ 5.,  5.],
        [-1., -1.],
        [ 0.,  0.],
        [-1., -1.],
        [ 0.,  0.],
        [ 0.,  0.],
        [-1., -1.],
        [ 0.,  0.]]
    )
    with torch.no_grad():
      expected = torch.mm(a, b)
      actual = QTensor.quantize(a).mm(QTensor.quantize(b))
    cmp_res = torch.allclose(other=expected, input=actual.dequantize(),atol=calc_max_mm_atol(QTensor.quantize(a), QTensor.quantize(b)))
    self.assertTrue(cmp_res)

  def test_const2(self):
    a = torch.tensor([
        [ 6.,  7.,  8., 10., 11., 12., 10., 11., 12., 14., 15., 16.,  6.,  7.,
          8., 10., 11., 12., 10., 11., 12., 14., 15., 16.]
    ])
    b = torch.tensor(
       [[ 0.],
        [-1.],
        [ 0.],
        [-1.],
        [ 5.],
        [-1.],
        [ 0.],
        [-1.],
        [ 0.],
        [ 0.],
        [-1.],
        [ 0.],
        [ 0.],
        [-1.],
        [ 0.],
        [-1.],
        [ 5.],
        [-1.],
        [ 0.],
        [-1.],
        [ 0.],
        [ 0.],
        [-1.],
        [ 0.]]
    )
    with torch.no_grad():
      expected = torch.mm(a, b)
      actual = QTensor.quantize(a).mm(QTensor.quantize(b))
    cmp_res = torch.allclose(other=expected, input=actual.dequantize(),atol=calc_max_mm_atol(QTensor.quantize(a), QTensor.quantize(b)))
    self.assertTrue(cmp_res)

  def test_random(self):
    h1_dim = random.randint(2, 2)
    w1_dim = random.randint(2, 2)
    w2_dim = random.randint(2, 2)
    maxint = int(2 ** 8 - 1)
    a = torch.randint(maxint, (h1_dim, w1_dim), dtype=torch.uint8, requires_grad=False)
    b = torch.randint(maxint, (w1_dim, w2_dim), dtype=torch.uint8, requires_grad=False)
    logger.debug(f"a.shape:{a.shape}, b.shape:{b.shape}")
    with torch.no_grad():
      expected = torch.mm(a.to(torch.int32), b.to(torch.int32))
      actual = quantize_tensor(a).mm(quantize_tensor(b))
      cmp_res = torch.allclose(other=expected.to(torch.float32), input=dequantize_tensor(actual),atol=calc_max_mm_atol(QTensor.quantize(a), QTensor.quantize(b)))
    self.assertTrue(cmp_res)

  def test_random_float(self):
    h1_dim = random.randint(30, 30)
    w1_dim = random.randint(30, 30)
    w2_dim = random.randint(30, 30)
    a = torch.rand(h1_dim, w1_dim, requires_grad=False)
    b = torch.rand(w1_dim, w2_dim, requires_grad=False)
    logger.debug(f"a.shape:{a.shape}, b.shape:{b.shape}")
    with torch.no_grad():
      expected = torch.mm(a, b)
      actual = quantize_tensor(a).mm(quantize_tensor(b))
      cmp_res = torch.allclose(other=expected, input=dequantize_tensor(actual),atol=calc_max_mm_atol(QTensor.quantize(a), QTensor.quantize(b)))
    self.assertTrue(cmp_res)

class TestMatmulQuant(unittest.TestCase):
  def test_2x2(self):
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8, requires_grad=False)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.uint8, requires_grad=False)
    with torch.no_grad():
      expected = a @ b
      actual = torch.matmul(QTensor.quantize(a),QTensor.quantize(b))
    cmp_res = torch.allclose(other=expected.to(torch.float32), input=actual.dequantize())
    self.assertTrue(cmp_res)

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

    q_x = QTensor.quantize(x)
    q_weight = QTensor.quantize(weight)
    q_bias = QTensor.quantize(bias, scale=(q_x.scale*q_weight.scale), zero_point=0, dtype=torch.int32)

    q_y = F.conv2d(q_x, weight=q_weight, bias=q_bias)
    self.assertTrue(torch.all(torch.eq(dequantize_tensor(q_y), torch.tensor([[[17.]]]))))

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
    self.ITER_NUM = 100
    self.randomizer = Conv2dRandomizer()

  def test_weight_only(self):
    with torch.no_grad():
      expected = F.conv2d(self.x, weight=self.weight)
      q_input = QTensor.quantize(self.x)
      q_weight = QTensor.quantize(self.weight)
      actual = F.conv2d(q_input, q_weight).dequantize()

    self.assertTrue(torch.allclose(input=actual, other=expected, atol=calc_max_conv2d_atol(input=q_input, weight=q_weight)))

  def test_bias(self):
    with torch.no_grad():
      expected = F.conv2d(self.x, weight=self.weight, bias=self.bias)
      q_input = QTensor.quantize(self.x)
      q_weight = QTensor.quantize(self.weight)
      q_bias = QTensor.quantize(self.bias, scale=(q_input.scale*q_weight.scale), zero_point=0, dtype=torch.int32)
      actual = F.conv2d(q_input, q_weight, bias=q_bias).dequantize()

    self.assertTrue(torch.allclose(input=actual, other=expected, atol=calc_max_conv2d_atol(input=q_input, weight=q_weight, bias=q_bias)))

class TestRandom(unittest.TestCase):
  def setUp(self):
    self.randomizer = Conv2dRandomizer()
    self.ITER_NUM = 100
    self.max_metrics: dict[Metrics, float] = {Metrics.ABS_ERROR: 0, Metrics.REL_ERROR: 0}

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
    with torch.no_grad():
      expected: torch.Tensor=self.call(input=input, weight=weight, bias=bias)
      q_input = QTensor.quantize(input, torch.int8)
      q_weight = QTensor.quantize(weight, torch.int8)
      q_bias = QTensor.quantize(bias, scale=(q_input.scale*q_weight.scale), zero_point=0, dtype=torch.int32)
      actual: torch.Tensor=self.call(input=q_input, weight=q_weight, bias=q_bias).dequantize()
    metrics = eval_metrics(actual=actual, expected=expected)
    metrics[Metrics.ATOL]=calc_max_conv2d_atol(input=q_input, weight=q_weight, bias=q_bias, conv2d=self.call)
    logger.debug(metrics2str(metrics))
    self.max_metrics[Metrics.ABS_ERROR] = max(self.max_metrics[Metrics.ABS_ERROR], metrics[Metrics.ABS_ERROR])
    self.max_metrics[Metrics.REL_ERROR] = max(self.max_metrics[Metrics.REL_ERROR], metrics[Metrics.REL_ERROR])
    # self.assertGreaterEqual(metrics[Metrics.MAX_EXPECTED], metrics[Metrics.ATOL]) # sanity check
    self.assertTrue(torch.allclose(input=actual, other=expected, atol=metrics[Metrics.ATOL]))

  def test_run(self):
    for i in range(self.ITER_NUM):
      with self.subTest(i=i): self.run_iteration()

if __name__ == '__main__':
  unittest.main()
