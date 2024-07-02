from typing import Callable, Sequence, Iterable, Any
import torch
import logging
from broquant.q_conv2d import q_conv2d
from broquant.q_linear import q_linear
from broquant.q_matmul import q_matmul
from broquant.utils import Implements, collapse_tensors

logger = logging.getLogger(__name__)

_HANDLED_FUNCTIONS = {}

implements = Implements(HANDLED_FUNCTIONS=_HANDLED_FUNCTIONS)

def dtype2min_max(dtype)->tuple[int, int]:
  return torch.iinfo(dtype).min, torch.iinfo(dtype).max

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
    return torch.mm(self, mat2)

  def __mul__(self, other: "QTensor") -> "QTensor":
    return torch.mul(self, other)

  def __imul__(self, other: "QTensor") -> "QTensor":
    return self.mul_(other)

  def mul_(self, other: "QTensor") -> "QTensor":
    self = torch.mul(self, other)
    return self

  def mul(self, other: "QTensor") -> "QTensor":
    return torch.mul(self, other)

  def __matmul__(self, other):
    return torch.matmul(self, other)

  def matmul(self, other):
    return torch.matmul(self, other)

  def __imatmul__(self, other: "QTensor") -> "QTensor":
    return self.matmul_(other)

  def matmul_(self, other: "QTensor") -> "QTensor":
    self = torch.matmul(self, other)
    return self

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if not all(issubclass(t, QTensor) for t in types):
      return NotImplemented
    if func not in _HANDLED_FUNCTIONS:
      if __debug__:
        tensors = [tensor for tensor in collapse_tensors(args) if isinstance(tensor, QTensor)]
        def is_same_scale_zp(tensors: Sequence) -> bool:
          return all((tensors[0].scale == tensor.scale) and (tensors[0].zero_point == tensor.zero_point) for tensor in tensors)
        def is_same_dtype(tensors: Sequence) -> bool:
          return all(tensors[0].dtype is tensor.dtype for tensor in tensors)
        if (len(tensors) > 1):
          if not is_same_scale_zp(tensors):
            raise NotImplementedError(f'QTensors have different scale/zp in func:{func}. Provide a specific handler if want to apply the func.')
          if not is_same_dtype(tensors):
            raise ValueError('tensors have different dtypes. You should requantize/cast types.')
      res = super().__torch_function__(func, types, args, kwargs)
      if isinstance(res, QTensor):
        arg = args[0][0] if type(args[0]) == list else args[0] # args[0] is list of QTensor for the func stack()
        res = arg.clone(new_tensor=res)
      return res
    return _HANDLED_FUNCTIONS[func](*args, **kwargs)

  @classmethod
  def quantize(cls, x: torch.Tensor, dtype=torch.uint8, min_val=None, max_val=None, scale=None, zero_point=0, zp_dtype=torch.int32):
    assert(isinstance(x, torch.Tensor))
    return quantize_tensor(x=x, dtype=dtype, min_val=min_val, max_val=max_val, scale=scale, zero_point=zero_point)

  def dequantize(self)->torch.Tensor:
    return dequantize_tensor(self)

def calcScaleZeroPoint(min_val, max_val, qmin, qmax, zp_min, zp_max)->tuple[float, int]:
  if min_val != max_val: # do the min-max quantization with the bias and the activation
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = round(qmin - min_val / scale)

    def clamp(n, smallest, largest): return max(smallest, min(n, largest))

    if not (zp_min <= zero_point <= zp_max):
      zero_point = clamp(zero_point, smallest=zp_min, largest=zp_max)
      scale = min_val / (qmin - zero_point) # recalc scale based on zp because we clamped zp

  else: # do the nearest scale quantization without the bias
    val = min_val
    min_zero_scaled = qmin
    max_zero_scaled = qmax
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

def quantize_tensor(x: torch.Tensor, dtype=torch.uint8, min_val=None, max_val=None, scale=None, zero_point=0, zp_dtype=torch.int32)->QTensor:

    qmin, qmax = dtype2min_max(dtype)

    zp_min, zp_max = dtype2min_max(zp_dtype)

    if not min_val and not max_val:
      min_val, max_val = x.min(), x.max()

    if not scale:
      scale, zero_point = calcScaleZeroPoint(min_val.item(), max_val.item(), qmin=qmin, qmax=qmax, zp_min=zp_min, zp_max=zp_max)
    q_x = zero_point + x / scale
    q_x.round_().clamp_(qmin, qmax)
    q_x = q_x.to(dtype)

    assert zp_min <= zero_point <= zp_max, f'zero_point ({zero_point}) is outside target range ([{zp_min}; {zp_max}])'

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x: QTensor)->torch.Tensor:
    return q_x.scale * (torch.Tensor(q_x).float() - q_x.zero_point)

@implements(torch.mul)
def q_mul(input: QTensor, other: QTensor):

  # see hints regarding the algorithm at https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  # Algorithm:
  # 1. x_n = x - zp: remove the bias of a tensor. To do so we need to increase the data storage from 8 bits up to int16. We choose a signed type for easy multiplication handling. It is unimportant whether the x is signed or unsigned because the zp already had been properly modified by a caller. Thus converts the tensor to the standard fixed-point format.
  # 2. y = x1 * x2, s = s1 * s2: multiply two fixed-point tensors. Increase data storage from int16 up to int32. Resulting bias is 0.
  # 3. Return y. Caller then can requantize and rescaled back to 8 bits.

  def check_dtype(tensor: QTensor)->None:
    dtype = tensor.dtype
    if not ((dtype is torch.uint8) or (dtype is torch.int8)):
      raise NotImplemented(f"Unsupported dtype:{dtype}")

  if __debug__:
    check_dtype(input)
    check_dtype(other)

  def QTensor2Tensor32(tensor: QTensor)->torch.Tensor:
    return (torch.Tensor(tensor).to(torch.int16) - tensor.zero_point).to(torch.int32)

  result_tensor = QTensor2Tensor32(input) * QTensor2Tensor32(other)
  result_scale = input.scale * other.scale

  return QTensor(tensor=result_tensor, scale=result_scale, zero_point=0)

@implements(torch.mm)
def q_mm(input: QTensor, mat2: QTensor)->QTensor:

  # see hints regarding the algorithm at https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  # 1. In a cycle do the standard matrix x = row_a x column_b vector dot multiplication by the q_mul function. It returns an int32 vector with scale_a * scale_b
  # 2. y = sum(x[i]). Thus we sum int32 into int32 accumulator
  # 3. Return y. A caller can requantize if this is needed.
  ar,ac = input.shape
  br,bc = mat2.shape
  assert ac==br
  c = QTensor(tensor=torch.zeros(ar, bc, dtype=torch.int32), scale=input.scale * mat2.scale, zero_point=0)
  c_dtype=c.dtype
  for i in range(ar):
    c[i] = (input[i].unsqueeze(-1) * mat2).sum(dim=0,dtype=c_dtype)
  return c

implements(torch.matmul)(q_matmul)
implements(torch.nn.functional.conv2d)(q_conv2d)
implements(torch.nn.functional.linear)(q_linear)

@implements(torch.nn.functional.unfold)
def q_unfold(input: QTensor, *args, **kwargs):
  unf_inp = torch.Tensor(input).float() # unfold doesn't support int
  unf_inp -= input.zero_point # unbias because unfould will pad zeros
  unf_out = torch.nn.functional.unfold(unf_inp, *args, **kwargs) + input.zero_point # restore zero_point (bias)
  return input.clone(new_tensor=(unf_out).round().clamp(*dtype2min_max(input.dtype)).to(input.dtype)) # round and clamp are added prematurely

@implements(torch.nn.functional.fold)
def q_fold(input: QTensor, *args, **kwargs):
  return input.clone(new_tensor=torch.nn.functional.fold(torch.Tensor(input).float(), *args, **kwargs).to(input.dtype)) # fold doesn't support int

def act_warning(act: Callable):
  def decorator(input: QTensor, *args, **kwargs):
    if not ((input.dtype is torch.int32) and (input.zero_point == 0.)):
      logger.warning(f'input dtype is not int32 or zp != 0. func: {act} is working in simulated mode.')
    return act(input, *args, **kwargs)
  return decorator

@implements(torch.nn.functional.relu)
@act_warning
def q_relu(input: QTensor, inplace=False):
  x = input if inplace else input.clone()
  x[(torch.Tensor(x).float() - x.zero_point) < 0.] = x.zero_point
  return x

@implements(torch.nn.functional.hardswish)
@act_warning
def q_hardswish(input: QTensor, inplace=False):
  x = input if inplace else input.clone()
  dequant_x = x.dequantize()
  left_range = dequant_x <= -3.
  middle_range = torch.logical_and(dequant_x > -3., dequant_x < 3.)
  x[left_range] = x.zero_point
  x[middle_range] = (((torch.Tensor(x[middle_range]).float() * torch.Tensor(x[middle_range]).float()) * x.scale + (torch.Tensor(x[middle_range]).float() * 3.)) / 6.).round().clamp(min=torch.iinfo(x.dtype).min, max=torch.iinfo(x.dtype).max).to(x.dtype)
  return x
