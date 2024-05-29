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
    return calc_mm_atol(self, mat2)

  # def __mul__(self, other: "TolTensor") -> "TolTensor":
    #   return torch.mul(self, other)

  def __matmul__(self, other):
    return torch.matmul(self, other)

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
          mult_tol = a_float[i,:].abs() * b_quant_tol + b_float[:,j].abs() * a_quant_tol
          c[i,j] = (mult_tol).sum() # multiply all of column j by all of row i and sum it
  return c
