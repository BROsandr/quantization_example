import sys
from broquant.Model import Model
import copy
import torch
from torchvision import datasets, transforms
from broquant.utils import test
from functools import partial
import torch.utils.data
from broquant.QModel import gatherStats, QModel
from pathlib import Path
from broquant.const import MNIST_MODEL_PATH, MNIST_DATASET_PATH
from math import isclose

import unittest

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class TestCmpLossAcc(unittest.TestCase):

  def test(self):
    dataset_path = MNIST_DATASET_PATH
    model = Model.create(load_model=True, model_path=MNIST_MODEL_PATH, dataset_path=dataset_path)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(str(dataset_path), train=False, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
      batch_size=64, shuffle=True, **kwargs)

    print('Model:')
    model_metrics = test(model, test_loader)

    q_model = copy.deepcopy(model)
    stats = gatherStats(q_model, test_loader)

    q_model = QModel(model=q_model, stats=stats)


    print(stats)

    print('QModel:')

    q_model_metrics = test(q_model, test_loader)

    self.assertTrue(isclose(model_metrics[1], q_model_metrics[1], rel_tol=0.1))

if __name__ == '__main__':

  unittest.main()
