import torch.nn.functional as F

from broquant.QTensor import quantize_tensor
from broquant.QTensor import calcScaleZeroPoint
from broquant.QTensor import dequantize_tensor
from broquant.QTensor import QTensor
import torch.nn as nn
from broquant.Model import Model
import torch

import logging
logger = logging.getLogger(__name__)

# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
  max_val, _ = torch.max(x, dim=1)
  min_val, _ = torch.min(x, dim=1)


  if key not in stats:
    stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
  else:
    stats[key]['max'] += max_val.sum().item()
    stats[key]['min'] += min_val.sum().item()
    stats[key]['total'] += 1

  return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

  x = F.relu(model.conv1(x))

  x = F.max_pool2d(x, 2, 2)

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')

  x = F.relu(model.conv2(x))

  x = F.max_pool2d(x, 2, 2)

  x = x.view(-1, 4*4*50)

  stats = updateStats(x, stats, 'fc1')

  x = F.relu(model.fc1(x))

  stats = updateStats(x, stats, 'fc2')

  x = model.fc2(x)

  return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cpu'

    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)

    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats

def quantize_parameters(model: Model):
  with torch.no_grad():
    for name, param in model.named_parameters():
      logger.debug(f'Quantizing param_name:{name}.')
      param.copy_(QTensor.quantize(param))

class QModel(nn.Module):
  def __init__(self, model: Model, stats):
    super().__init__()

    self.model = model
    self.stats = stats
    quantize_parameters(self.model)

  def forward(self, x: torch.Tensor):
    stats = self.stats
    model = self.model

    # Quantise before inputting into incoming layers
    q_x = QTensor.quantize(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

    q_x = model.conv1(q_x)

    def requantize(q_x: QTensor, stats) -> QTensor:
      return QTensor.quantize(q_x.dequantize(), min_val=stats['min'], max_val=stats['max'])

    q_x = requantize(q_x, stats['conv2'])

    q_x = F.max_pool2d(q_x, 2, 2)

    q_x = model.conv2(q_x)

    q_x = requantize(q_x, stats['fc1'])

    q_x = F.max_pool2d(q_x, 2, 2)

    q_x = q_x.view(-1, 4*4*50)

    q_x = model.fc1(q_x)

    q_x = requantize(q_x, stats['fc2'])

    # Back to dequant for final layer
    x = dequantize_tensor(q_x)

    x = model.fc2(x)

    return F.log_softmax(x, dim=1)
