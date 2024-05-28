from typing import Sequence, Iterable, Any
import torch
import logging
from broquant.q_conv2d import q_conv2d
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

    assert(self.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32))
    self.scale = scale
    self.zero_point = zero_point

  def clone(self, new_tensor=None, *args, **kwargs) -> "QTensor":
    if new_tensor is None:
      new_tensor = super().clone(*args, **kwargs)
    return QTensor(tensor=new_tensor, scale=self.scale, zero_point=self.zero_point)

  def mm(self, mat2: "QTensor") -> "QTensor":
    return q_mm(self, mat2)

  def __mul__(self, other: "QTensor") -> "QTensor":
    return q_mul(self, other)

  def __matmul__(self, other):
    return torch.matmul(self, other)

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if not all(issubclass(t, QTensor) for t in types):
      return NotImplemented
    if func not in _HANDLED_FUNCTIONS:
      tensors = [tensor for tensor in args if isinstance(tensor, QTensor)]
      def is_same_scale_zp(tensors: Sequence) -> bool:
        return all((tensors[0].scale == tensor.scale) and (tensors[0].zero_point == tensor.zero_point) for tensor in tensors)
      if (len(tensors) > 1) and (not is_same_scale_zp(tensors)):
        raise NotImplementedError(f'QTensors have different scale/zp in func:{func}. Provide a specific handler if want to apply the func.')
      res = super().__torch_function__(func, types, args, kwargs)
      if isinstance(res, QTensor):
        arg = args[0][0] if isinstance(args[0], list) else args[0] # args[0] is list of QTensor for the func stack()
        res = arg.clone(new_tensor=res)
      return res
    return _HANDLED_FUNCTIONS[func](*args, **kwargs)

  @classmethod
  def quantize(cls, x: torch.Tensor, num_bits=8, min_val=None, max_val=None):
    assert(isinstance(x, torch.Tensor))
    return quantize_tensor(x=x, num_bits=num_bits, min_val=min_val, max_val=max_val)

  def dequantize(self)->torch.Tensor:
    return dequantize_tensor(self)

def calcScaleZeroPoint(min_val, max_val,num_bits=8)->tuple[float, int]:
  # Calc Scale and zero point of next
  qmin = 0.
  qmax = 2 ** num_bits - 1.

  if min_val != max_val: # do the min-max quantization with the bias and the activation
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = round(qmin - min_val / scale)
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

    zero_point = round(-min_zero_scaled * scale) # convert to uint range. Note that here we use multiplication not division. The divison is used in the case of min_val.

  return scale, zero_point

def quantize_tensor(x: torch.Tensor, num_bits=8, min_val=None, max_val=None)->QTensor:

    if not min_val and not max_val:
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val.item(), max_val.item(), num_bits)
    q_x = zero_point + x / scale
    q_x.round_().clamp_(qmin, qmax)
    q_x = q_x.byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x: QTensor)->torch.Tensor:
    return q_x.scale * (torch.Tensor(q_x).float() - q_x.zero_point)

def q_mul(input: QTensor, other: QTensor):

  # see hints regarding the algorithm at https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  # Algorithm:
  # 1. x_n = x - zp: remove the bias from a tensor. To do so we need to increase the data storage from 8 bits up to int16. We choose a signed type for easy multiplication handling. It is unimportant whether the x is signed or unsigned because the zp already had been properly modified be a caller. Thus converts the tensor to the standard fixed-point format.
  # 2. y = x1 * x2, s = s1 * s2: multiply two fixed-point tensors. Increase data storage from int16 up to int32. Resulting bias is 0.
  # 3. Return y. Caller then can requantize and rescaled back to 8 bits.

  def check_dtype(tensor: QTensor)->None:
    dtype = tensor.dtype
    if not ((dtype is torch.uint8) or (dtype is torch.int8)):
      raise NotImplemented(f"Unsupported dtype:{dtype}")

  check_dtype(input)
  check_dtype(other)

  def QTensor2Tensor32(tensor: QTensor)->torch.Tensor:
    return (torch.Tensor(tensor).to(torch.int16) - tensor.zero_point).to(torch.int32)

  result_tensor = QTensor2Tensor32(input) * QTensor2Tensor32(other)
  result_scale = input.scale * other.scale

  return QTensor(tensor=result_tensor, scale=result_scale, zero_point=0)

def q_mm(input: QTensor, mat2: QTensor)->QTensor:

  # see hints regarding the algorithm at https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  # 1. In a cycle do the standard matrix x = row_a x column_b vector dot multiplication by the q_mul function. It returns an int32 vector with scale_a * scale_b
  # 2. y = sum(x[i]). Thus we sum int32 into int32 accumulator
  # 3. Return y. A caller can requantize if this is needed.
  ar,ac = input.shape
  br,bc = mat2.shape
  assert ac==br
  c = QTensor(tensor=torch.zeros(ar, bc, dtype=torch.int32), scale=input.scale * mat2.scale, zero_point=0)
  for i in range(ar):
      for j in range(bc):
          c[i,j] = (input[i,:] * mat2[:,j]).sum(dtype=c.dtype) # multiply all of column j by all of row i and sum it
  return c

@implements(torch.matmul)
def q_matmul(input: torch.Tensor, other: torch.Tensor)->torch.Tensor:
  """
    Partial implementation of torch.matmul. Utilizes torch.mm.
  """

  is_valid_shape_len = lambda tensor: 2 <= len(tensor.shape) <= 4
  if not(is_valid_shape_len(input) and is_valid_shape_len(other)):
    raise ValueError(f'Unsupported shape. input.shape:{input.shape}, other.shape:{other.shape}')

  def normalize_shape(tensor: torch.Tensor)->int:
    num_of_extra_dims = 0
    while len(tensor.shape) != 4:
      tensor.unsqueeze_(0) # prepend dimension
      num_of_extra_dims += 1
    return num_of_extra_dims

  def denormalize_shape(tensor: torch.Tensor, num_of_extra_dims)->None:
    for _ in range(num_of_extra_dims): tensor.squeeze_(0)

  inp_num_of_extra_dims = normalize_shape(input)
  other_num_of_extra_dims = normalize_shape(other)

  from functools import partial

  denormalize_shape = partial(denormalize_shape, num_of_extra_dims=min(inp_num_of_extra_dims, other_num_of_extra_dims))

  are_dims_compatible = lambda dim1, dim2: (dim1 == dim2) or (dim1 == 1 or dim2 == 1)
  assert(all([are_dims_compatible(inp_dim, other_dim) for inp_dim, other_dim in zip(input.shape, other.shape)][0:2]))

  def get_broadcast_indices(dim1: int, dim2: int)->Iterable[tuple[int, int]]:
    from itertools import zip_longest
    return zip_longest(range(dim1), range(dim2), fillvalue=0) # for 1-dims an index will always be 0

  batch_out: list[list[torch.Tensor]] = []
  for inp_batch_idx, other_batch_idx in get_broadcast_indices(input.shape[0], other.shape[0]):
    channel_out: list[torch.Tensor] = []
    for inp_chan_idx, other_chan_idx in get_broadcast_indices(input.shape[1], other.shape[1]):
      channel_out.append(input[inp_batch_idx, inp_chan_idx, ...].mm(other[other_batch_idx, other_chan_idx, ...]))
    batch_out.append(channel_out)

  def cat_tensors(seq: tuple | list)->torch.Tensor:
    if all(isinstance(el, torch.Tensor) for el in seq):
      return torch.stack(seq)
    return cat_tensors([cat_tensors(el) for el in seq])

  out = cat_tensors(batch_out)
  denormalize_shape(out)

  return out

implements(torch.nn.functional.conv2d)(q_conv2d)
