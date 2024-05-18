import torch.nn.functional as F

from QTensor import quantize_tensor
from QTensor import calcScaleZeroPoint

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

  return x, scale_next, zero_point_next
