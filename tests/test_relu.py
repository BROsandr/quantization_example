import unittest
import sys
import torch
from torch.nn import functional as F
import torch.nn as nn
import random
import os
from broquant.utils import Metrics, eval_metrics, metrics2str

from broquant.QTensor import QTensor
from broquant.TolTensor import TolTensor

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

class ReluRandomizer:
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.MAX_RAND = 10

  def randomize(self):
    dim_w = random.randint(1, self.MAX_RAND)
    dim_h = random.randint(1, self.MAX_RAND)
    self.in_channels = random.randint(1, self.MAX_RAND)
    num_batches = random.randint(1,self.MAX_RAND )
    self.input = torch.rand(num_batches, self.in_channels, dim_h, dim_w, requires_grad=False)

def calc_max_relu_atol(input: QTensor, func=F.relu)->float:
  input_tol_tensor = TolTensor.from_QTensor(input)
  return func(input=input_tol_tensor).atol

class TestConst(unittest.TestCase):
  def test_pos_neg(self):
    x = torch.tensor([-2., 1], requires_grad=False)
    with torch.no_grad():
      expected = torch.tensor([0., 1], requires_grad=False)
      q_input = QTensor.quantize(x, dtype=torch.int8)
      actual = F.relu(q_input).dequantize()
    cmp_res = torch.allclose(expected, actual, atol=calc_max_relu_atol(input=q_input))
    self.assertTrue(cmp_res)

class TestRandom(unittest.TestCase):
  def setUp(self):
    self.randomizer = ReluRandomizer()
    self.ITER_NUM = 100
    self.max_metrics: dict[Metrics, float] = {Metrics.ABS_ERROR: 0, Metrics.REL_ERROR: 0}

  def call(self, input):
    return F.relu(input=input)

  def run_iteration(self):
    randomizer = self.randomizer
    randomizer.randomize()
    input = randomizer.input
    with torch.no_grad():
      expected: torch.Tensor=self.call(input=input)
      q_input = QTensor.quantize(input)
      actual: torch.Tensor=self.call(input=q_input).dequantize()
    metrics = eval_metrics(actual=actual, expected=expected)
    metrics[Metrics.ATOL]=calc_max_relu_atol(input=q_input, func=self.call)
    logger.debug(metrics2str(metrics))
    self.max_metrics[Metrics.ABS_ERROR] = max(self.max_metrics[Metrics.ABS_ERROR], metrics[Metrics.ABS_ERROR])
    self.max_metrics[Metrics.REL_ERROR] = max(self.max_metrics[Metrics.REL_ERROR], metrics[Metrics.REL_ERROR])
    self.assertGreaterEqual(metrics[Metrics.MAX_EXPECTED], metrics[Metrics.ATOL]) # sanity check
    self.assertTrue(torch.allclose(input=actual, other=expected, atol=metrics[Metrics.ATOL]))

  def test_run(self):
    for i in range(self.ITER_NUM):
      with self.subTest(i=i): self.run_iteration()
    logger.info(f'''For all iteration.\n{metrics2str({
        Metrics.ABS_ERROR: self.max_metrics[Metrics.ABS_ERROR],
        Metrics.REL_ERROR: self.max_metrics[Metrics.REL_ERROR]
    })}''')

if __name__ == '__main__':
  unittest.main()
