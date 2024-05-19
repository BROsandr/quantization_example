import torch

_HANDLED_FUNCTIONS = {}

import functools
def implements(torch_function):
  """Register a torch function override for QTensor"""
  def decorator(func):
      functools.update_wrapper(func, torch_function)
      _HANDLED_FUNCTIONS[torch_function] = func
      return func
  return decorator

class QTensor():
  def __init__(self, tensor: torch.Tensor, scale: float, zero_point: int = 0):
    super().__init__()

    self.tensor = tensor
    self.scale = scale
    self.zero_point = zero_point

  @property
  def tensor(self):
    return self._tensor

  @tensor.setter
  def tensor(self, new_data: torch.Tensor):
    assert(new_data.dtype in (torch.uint8, torch.int8))
    self._tensor = new_data

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if (func not in _HANDLED_FUNCTIONS) or (not all(issubclass(t, QTensor) for t in types)):
      return NotImplemented
    return _HANDLED_FUNCTIONS[func](*args, **kwargs)

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
