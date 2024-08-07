import unittest
import torch
from broquant.QTensor import dequantize_tensor, quantize_tensor
import random

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class TestQuantize(unittest.TestCase):
  def test_quantize_2x2(self):
    x = torch.tensor([[133,  70],
          [ 68,  67]], dtype=torch.uint8)

    q_x = quantize_tensor(x)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x.float(), atol=q_x.scale/2))

  def test_quantize_1x1(self):
    x = torch.tensor([123], dtype=torch.uint8)

    q_x = quantize_tensor(x)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x.float(), atol=q_x.scale/2))

  def test_quantize_1x1_neg(self):
    x = torch.tensor([-61.])

    with self.assertRaises(ValueError):
      q_x = quantize_tensor(x, dtype=torch.uint8)

    q_x = quantize_tensor(x.float(), dtype=torch.int8)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x.float(), atol=q_x.scale/2))

  def test_quantize_int16(self):
    x = torch.tensor([-320.])

    q_x = quantize_tensor(x.float(), dtype=torch.int16)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x.float(), atol=q_x.scale/2))

  def test_two_int16(self):
    x = torch.tensor([-320., 161])

    q_x = quantize_tensor(x.float(), dtype=torch.int16)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x.float(), atol=q_x.scale/2))

  def test_rand(self):
    w_dim = random.randint(1, 100)
    h_dim = random.randint(1, 100)
    maxint = int(2 ** 8 - 1)
    x = torch.randint(maxint, (h_dim, w_dim), dtype=torch.uint8, requires_grad=False)
    logger.debug(f"x.shape:{x.shape}")
    q_x = quantize_tensor(x)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x.float(), atol=q_x.scale/2))

  def test_rand_float(self):
    w_dim = random.randint(1, 100)
    h_dim = random.randint(1, 100)
    x = torch.rand((h_dim, w_dim), requires_grad=False)
    logger.debug(f"x.shape:{x.shape}")
    q_x = quantize_tensor(x)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x, atol=q_x.scale/2))

  def test_zp_out_of_int8(self):
    x = torch.tensor([0.8832, 0.7181], requires_grad=False)
    q_x = quantize_tensor(x, zp_dtype=torch.int8)
    self.assertTrue(torch.iinfo(torch.int8).min <= q_x.zero_point <= torch.iinfo(torch.int8).max)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x, atol=q_x.scale/2))

  def test_zp_not_zero(self):
    x = torch.tensor([2., 3.], requires_grad=False)
    q_x = quantize_tensor(x, dtype=torch.int8)
    self.assertNotEqual(q_x.zero_point, 0)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x, atol=q_x.scale/2))

  def test_zp_not_zero2(self):
    x = torch.tensor([-2., 1], requires_grad=False)
    q_x = quantize_tensor(x, dtype=torch.int8, zp_dtype=torch.int8)
    self.assertNotEqual(q_x.zero_point, 0)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x, atol=q_x.scale/2))

  def test_127(self):
    x = torch.tensor([127.], requires_grad=False)
    q_x = quantize_tensor(x, dtype=torch.int32)
    self.assertTrue(torch.allclose(dequantize_tensor(q_x), x, atol=q_x.scale/2))

if __name__ == '__main__':
  unittest.main()
