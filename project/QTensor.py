from typing import NamedTuple
import torch

class QTensor(NamedTuple):
  scale: float
  tensor: torch.Tensor
  zero_point: int = 0

def calcScaleZeroPoint(min_val, max_val,num_bits=8)->tuple[float, int]:
  # Calc Scale and zero point of next
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  zero_point = int(qmin - min_val / scale)

  return scale, zero_point

def quantize_tensor(x: torch.Tensor, num_bits=8, min_val=None, max_val=None)->QTensor:

    if not min_val and not max_val:
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x: QTensor)->torch.Tensor:
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
