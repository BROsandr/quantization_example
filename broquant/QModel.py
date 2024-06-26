import torch.nn.functional as F

from broquant.QTensor import quantize_tensor
from broquant.QTensor import calcScaleZeroPoint
from broquant.QTensor import dequantize_tensor
from broquant.QTensor import QTensor, dtype2min_max
import torch.nn as nn
from broquant.Model import Model
import torch

import logging
logger = logging.getLogger(__name__)

# Get Min and max of x tensor, and stores it
def updateStats(x, stats: dict, key):
  max_val = x.max()
  min_val = x.min()

  if key not in stats:
    stats[key] = {"max": max_val, "min": min_val}
  else:
    stats[key]['max'] = max(stats[key]['max'], max_val)
    stats[key]['min'] = min(stats[key]['min'], min_val)

  return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

  x = F.relu(model.conv1(x))

  x = torch.flatten(x, start_dim=1)

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

    return stats

def quantize_bias(module: nn.Module, min_val, max_val):
  logger.debug('In quantize_bias(...).')
  dtype = torch.uint8
  qmin, qmax = dtype2min_max(dtype)
  act_scale, act_zp = calcScaleZeroPoint(min_val=min_val, max_val=max_val, qmin=qmin, qmax=qmax)
  module.bias = nn.Parameter(QTensor.quantize(module.bias, dtype=torch.int32, scale=(module.weight.scale * act_scale), zero_point=0), requires_grad=False)

def quantize_parameters(model: Model, stats):
  with torch.no_grad():
    for module_name, module in model.named_modules():
      if type(module) in (nn.Conv2d, nn.Linear):
        if module_name == 'fc2':
          logger.debug(f'Skipping quantization of module:{module_name}.')
          continue
        for param_name, param in module.named_parameters():
          logger.debug(f'Quantizing param_name:{param_name} in module {module_name}.')
          if param_name == 'bias':
            quantize_bias(module=module, min_val=stats[module_name]['min'].item(), max_val=stats[module_name]['max'].item())
          else:
            setattr(module, param_name, nn.Parameter(QTensor.quantize(param), requires_grad=False))

class QModel(nn.Module):
  def __init__(self, model: Model, stats):
    super().__init__()

    self.model = model
    self.stats = stats
    quantize_parameters(model=self.model, stats=stats)

  def forward(self, x: torch.Tensor):
    stats = self.stats
    model = self.model

    # Quantise before inputting into incoming layers
    q_x = QTensor.quantize(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

    q_x = model.conv1(q_x)

    def requantize(q_x: QTensor, stats) -> QTensor:
      return QTensor.quantize(q_x.dequantize(), min_val=stats['min'], max_val=stats['max'])

    q_x = requantize(q_x, stats['fc1'])

    q_x = torch.flatten(q_x, start_dim=1)

    q_x = model.fc1(q_x)

    q_x = requantize(q_x, stats['fc2'])

    # Back to dequant for final layer
    x = dequantize_tensor(q_x)

    x = model.fc2(x)

    return F.log_softmax(x, dim=1)
