import torch
import logging
from broquant.QTensor import implements, QTensor
from broquant.utils import Implements

logger = logging

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

  def __mul__(self, other: "TolTensor") -> "TolTensor":
    return torch.mul(self, other)

  def mul(self, other: "TolTensor") -> "TolTensor":
    return torch.mul(self, other)

  def __add__(self, other):
    return torch.add(self, other)

  def add(self, other):
    return torch.add(self, other)

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

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if not all(issubclass(t, TolTensor) for t in types):
      return NotImplemented
    if func not in _HANDLED_FUNCTIONS:
      res = super().__torch_function__(func, types, args, kwargs)
      if isinstance(res, TolTensor):
        arg = args[0][0] if isinstance(args[0], list) else args[0] # args[0] is list of QTensor for the func stack()
        res = arg.clone(new_tensor=res)
      return res
    return _HANDLED_FUNCTIONS[func](*args, **kwargs)

  @classmethod
  def from_QTensor(cls, x: QTensor, dtype=torch.uint8, min_val=None, max_val=None, scale=None, zero_point=0):
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

  a_float = a
  b_float = b

  a_quant_tol = a.atol
  b_quant_tol = b.atol

  c = torch.zeros(ar, bc)
  for i in range(ar):
      for j in range(bc):
          mult_tol = a_float[i,:] * b_float[:,j]
          c[i,j] = (mult_tol).sum() # multiply all of column j by all of row i and sum it
  return c

@implements(torch.mul)
def tol_tensor_mul(input, other):
  """
    Returns a mul result with atol calculated as maximum among all element product's tolerances.
  """
  tensor_atol = None
  tensor_res = None
  if isinstance(other, TolTensor):
    tensor_atol = input.atol * torch.Tensor(other).abs() + other.atol * torch.Tensor(input).abs()
    tensor_res = torch.Tensor(other)
  res_atol = tensor_atol.max().item() if tensor_atol is not None else input.atol * other
  res = torch.Tensor(input) * (tensor_res if tensor_res is not None else other)
  return TolTensor(tensor=res, atol=res_atol)

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
