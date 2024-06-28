import torch
import logging
from broquant.QTensor import implements, QTensor, q_matmul
from broquant.utils import Implements, collapse_tensors
from broquant.q_conv2d import q_conv2d
from broquant.q_linear import q_linear

logger = logging.getLogger(__name__)

_HANDLED_FUNCTIONS = {}

implements = Implements(HANDLED_FUNCTIONS=_HANDLED_FUNCTIONS)

class TolTensor(torch.Tensor):
  def __new__(cls, tensor: torch.Tensor, atol: float, *args, **kwargs):
    return super().__new__(cls, tensor, *args, **kwargs)

  def __init__(self, tensor: torch.Tensor, atol: float,):
    super().__init__()

    self.atol = atol

  def clone(self, new_tensor=None, *args, **kwargs) -> "TolTensor":
    if new_tensor is None:
      new_tensor = super().clone(*args, **kwargs)
    return TolTensor(tensor=new_tensor, atol=self.atol)

  def mm(self, mat2: "TolTensor") -> "TolTensor":
    return torch.mm(self, mat2)

  def __mul__(self, other) -> "TolTensor":
    return torch.mul(self, other)

  def mul(self, other) -> "TolTensor":
    return torch.mul(self, other)

  def __truediv__(self, other) -> "TolTensor":
    return torch.div(self, other)

  def div(self, other) -> "TolTensor":
    return torch.div(self, other)

  def __add__(self, other):
    return torch.add(self, other)

  def add(self, other):
    return torch.add(self, other)

  def mul_(self, other) -> "TolTensor":
    self = torch.mul(self, other)
    return self

  def div_(self, other) -> "TolTensor":
    self = torch.div(self, other)
    return self

  def __imul__(self, other) -> "TolTensor":
    return self.mul_(other)

  def __itruediv__(self, other) -> "TolTensor":
    return self.div_(other)

  def __iadd__(self, other):
    return self.add_(other)

  def __isub__(self, other):
    return self.sub_(other)

  def matmul_(self, other):
    self = torch.matmul(self, other)
    return self

  def add_(self, other):
    self = torch.add(self, other)
    return self

  def sub_(self, other):
    self = torch.sub(self, other)
    return self

  def __sub__(self, other):
    return torch.add(self, -other)

  def sub(self, other):
    return torch.add(self, -other)

  def __matmul__(self, other):
    return torch.matmul(self, other)

  def matmul(self, other):
    return torch.matmul(self, other)

  def sum(self, dtype=None):
    return torch.sum(self, dtype=dtype)

  def __getitem__(self, key):
    if not((type(key) == type(self)) and (key.dtype == torch.bool)): return NotImplemented
    return super(TolTensor, self).__getitem__(torch.Tensor(key))

  def __setitem__(self, key, value):
    if not((type(key) == type(self)) and (key.dtype == torch.bool)): return NotImplemented
    if not(type(value) == type(self)): return NotImplemented
    super(TolTensor, self).__setitem__(torch.Tensor(key), torch.Tensor(value))

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if not all(issubclass(t, TolTensor) for t in types):
      return NotImplemented
    if func not in _HANDLED_FUNCTIONS:
      tensors = [tensor for tensor in collapse_tensors(args) if isinstance(tensor, TolTensor)]
      if len(tensors) > 1:
        raise NotImplementedError(f'Multiple TolTensor arguments are passed in func:{func}. Provide a custom implementation for their atol handling.')
      res = super().__torch_function__(func, types, args, kwargs)
      if isinstance(res, TolTensor):
        arg = tensors[0] # args[0] is list of QTensor for the func stack()
        res = arg.clone(new_tensor=res)
      return res
    return _HANDLED_FUNCTIONS[func](*args, **kwargs)

  @classmethod
  def from_QTensor(cls, x: QTensor):
    assert(isinstance(x, QTensor))
    return TolTensor(tensor=x.dequantize(), atol=x.scale/2)

@implements(torch.mm)
def calc_mm_atol(a, b)->TolTensor:
  """
    Calculates absolute tolerances of the corresponding element of a matrix multiplication result and returns a matrix with the tolerances.

    see formulas at https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
  """

  # mm = sum(a*b)
  ar,ac = a.shape
  br,bc = b.shape
  assert ac==br
  c = torch.zeros(ar, bc)
  max_atol = 0.
  for i in range(ar):
      for j in range(bc):
          el = (a[i,:] * b[:,j]).sum() # multiply all of column j by all of row i and sum it
          max_atol = max(max_atol, el.atol)
          c[i,j] = el
  return TolTensor(tensor=c, atol=max_atol)

@implements(torch.mul)
def tol_tensor_mul(input, other):
  """
    Returns a mul result with atol calculated as maximum among all element product's tolerances.
  """
  tensor_atol = None
  tensor_res = None
  res_atol = None
  if isinstance(other, TolTensor):
    tensor_res = torch.Tensor(input) * torch.Tensor(other)
    tensor_atol = tensor_res * (input.atol / torch.Tensor(input).abs() + other.atol / torch.Tensor(other).abs())
    res_atol = tensor_atol.max().item()
  else:
    tensor_res = torch.Tensor(input) * other
    res_atol = input.atol * other
  return TolTensor(tensor=tensor_res, atol=res_atol)

@implements(torch.div)
def tol_tensor_div(input, other):
  divider = None
  if isinstance(other, torch.Tensor):
    divider = other.clone(1. / torch.Tensor(other))
  else:
    divider = 1. / other
  return torch.mul(input, divider)

@implements(torch.sub)
def tol_tensor_sub(input, other):
  return torch.add(input, -other)

@implements(torch.add)
def tol_tensor_add(input, other):
  """
    Returns an add result with atol calculated as sum of input's atol and other's atol (if other is tensor).
  """
  tensor_atol = None
  tensor_res = None
  if isinstance(other, TolTensor):
    tensor_atol = input.atol + other.atol
    tensor_res = torch.Tensor(other)
  res_atol = tensor_atol if tensor_atol is not None else input.atol
  res = torch.Tensor(input) + (tensor_res if tensor_res is not None else other)
  return TolTensor(tensor=res, atol=res_atol)

@implements(torch.sum)
def tol_tensor_sum(input, dtype=None):
  """
    Returns a sum result with atol calculated as sum of every element's atol in input.
  """
  res = torch.Tensor(input).sum(dtype=dtype)
  res_atol = input.numel() * input.atol
  return TolTensor(tensor=res, atol=res_atol)

@implements(torch.stack)
def tol_tensor_stack(tensors, *args, **kwargs):
  assert(all(isinstance(tensor, TolTensor) for tensor in tensors))
  tensor_seq = [torch.Tensor(tensor) for tensor in tensors]
  res = torch.stack(tensor_seq, *args, **kwargs)
  res_atol = max(tensor.atol for tensor in tensors)
  return TolTensor(tensor=res, atol=res_atol)

implements(torch.nn.functional.conv2d)(q_conv2d)
implements(torch.nn.functional.linear)(q_linear)
implements(torch.matmul)(q_matmul)

@implements(torch.nn.functional.fold)
def tol_fold(input, *args, **kwargs):
  res=torch.nn.functional.fold(torch.Tensor(input), *args, **kwargs) # fold doesn't support int
  if input.numel() != res.numel():
    raise NotImplementedError("Number of output's elements differs from the input's. Implement atol recalculation.")
  return TolTensor(tensor=res, atol=input.atol)

@implements(torch.nn.functional.hardswish)
def tol_hardswish(input: TolTensor, inplace=False):
  x = input if inplace else input.clone()
  left_range = x <= -3.
  middle_range = torch.logical_and(x > -3., x < 3.)
  x[left_range] = 0
  x[middle_range] = ((x[middle_range] * x[middle_range]) + (x[middle_range] * 3.)) / 6.
  return x

@implements(torch.logical_and)
def tol_logical_and(input: TolTensor, other: TolTensor):
  res = torch.logical_and(torch.Tensor(input), torch.Tensor(input))
  res_atol = 0
  return TolTensor(tensor=res, atol=res_atol)
