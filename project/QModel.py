import torch.nn.functional as F

from QTensor import quantize_tensor
from QTensor import calcScaleZeroPoint
from QTensor import dequantize_tensor
from QTensor import QTensor
import torch.nn as nn
from Model import Model
import torch

def quantizeLayer(x, layer, stat, scale_x, zp_x):
  # for both conv and linear layers

  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # quantise weights, activations are already quantised
  w = quantize_tensor(layer.weight.data)
  b = quantize_tensor(layer.bias.data)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  # This is Quantisation Artihmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point

  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

  # Preparing input by shifting
  X = x.float() - zp_x
  layer.weight.data = scale_x * scale_w*(layer.weight.data - zp_w)
  layer.bias.data = scale_b*(layer.bias.data + zp_b)

  # All int computation
  x = (layer(X)/ scale_next) + zero_point_next

  # Perform relu too
  x = F.relu(x)

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B

  return x.round().byte(), scale_next, zero_point_next

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

class QModel(nn.Module):
  def __init__(self, model: Model, stats):
    super().__init__()

    self.model = model
    self.stats = stats

  def forward(self, x: torch.Tensor):
    stats = self.stats
    model = self.model

    # Quantise before inputting into incoming layers
    x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

    x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point)

    x = F.max_pool2d(x, 2, 2)

    x, scale_next, zero_point_next = quantizeLayer(x, model.conv2, stats['fc1'], scale_next, zero_point_next)

    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 4*4*50)

    x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'], scale_next, zero_point_next)

    # Back to dequant for final layer
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

    x = model.fc2(x)

    return F.log_softmax(x, dim=1)

if __name__ == '__main__':
  from Model import Model
  import copy
  import torch
  from torchvision import datasets, transforms
  from utils import test
  from functools import partial

  model = Model.create()

  kwargs = {'num_workers': 1, 'pin_memory': True}
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs)

  print('Model:')
  test(model, test_loader)

  q_model = copy.deepcopy(model)
  stats = gatherStats(q_model, test_loader)

  q_model = QModel(model=q_model, stats=stats)


  print(stats)

  print('QModel:')

  test(q_model, test_loader)
