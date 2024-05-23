import torch
import logging
logger = logging

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
  qmax = 2 ** num_bits - 1.

  if min_val != max_val: # do the min-max quantization with the bias and the activation
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = int(qmin - min_val / scale)
  else: # do the nearest scale quantization without the bias
    val = min_val
    min_zero_scaled = (-1 << (num_bits - 1))
    max_zero_scaled = ( 1 << (num_bits - 1)) - 1
    if not (min_zero_scaled <= val <= max_zero_scaled):
      raise ValueError(f"A quantized value is outside range [{min_zero_scaled}; {max_zero_scaled}], when min_val == max_val")
    logger.warning("Qunatizing a value when min_val == max_val. This is a low precision mode.")

    def find_scale(val)->float:
      scale = 1
      if int(val) == val: # if the val is int then it is excactly representable. Also handles 0 case.
        return scale # so do shortcut

      while (min_zero_scaled * scale) <= val <= (max_zero_scaled * scale): # Scale down the representable range until the val would be outside
        scale /= 2
      return scale * 2 # Because we did one iteration past the actual
    scale = find_scale(val)

    zero_point = int(-min_zero_scaled * scale) # convert to uint range. Note that here we use multiplication not division. The divison is used in the case of min_val.

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
