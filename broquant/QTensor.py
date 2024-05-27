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

class QTensor(torch.Tensor):
  def __new__(cls, tensor: torch.Tensor, scale: float, zero_point: int = 0, *args, **kwargs):
    return super().__new__(cls, tensor, *args, **kwargs)

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
    assert(self.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32))
    self._tensor = new_data

  def clone(self, new_tensor=None, *args, **kwargs) -> "QTensor":
    if new_tensor is None:
      new_tensor = super().clone(*args, **kwargs)
    return QTensor(tensor=new_tensor, scale=self.scale, zero_point=self.zero_point)

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if not all(issubclass(t, QTensor) for t in types):
      return NotImplemented
    if func not in _HANDLED_FUNCTIONS:
      def all_equal(iterable, key=None):
        from itertools import groupby
        g = groupby(iterable)
        return next(g, True) and not next(g, False)
      tensors = [tensor for tensor in args if isinstance(tensor, QTensor)]
      def is_same_scale_zp(tensors):
        return all_equal(tensors, lambda q1, q2: (q1.scale == q2.scale) and (q1.zero_point == q2.zero_point))
      if not is_same_scale_zp(tensors):
        return NotImplemented
      res = super().__torch_function__(func, types, args, kwargs)
      if isinstance(res, QTensor):
        res = QTensor(tensor=res, scale=tensors[0].scale, zero_point=tensors[0].zero_point)
      return res
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
