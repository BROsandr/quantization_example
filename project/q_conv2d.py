from torch.nn.functional import conv2d
from .QTensor import implements

@implements(conv2d)
def q_conv2d(input, weight, bias=None, stride=1, padding=0):
  from .QTensor import quantize_tensor, dequantize_tensor
  input = dequantize_tensor(input)
  weight = dequantize_tensor(weight)
  if bias is not None:
    bias = dequantize_tensor(bias)
  return quantize_tensor(conv2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding))
