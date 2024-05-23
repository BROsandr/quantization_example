from torch.nn.functional import conv2d
from broquant.QTensor import implements

@implements(conv2d)
def q_conv2d(input, weight, bias=None, stride=1, padding=0):
  # TODO:
  # Algorithm:
  # 1. conv2d = unfold - matmul - fold
  # 2. unfold, fold: keep the pytorch's implementation because these operations are data manipulations.
  # 3. matmul will yield int32 with scale = s1 * s2. Fold will do channel-wise addition of the matmul's result (int32 + int32). Then we add to the fold's result the bias, quantized with the scale = s1 * s2 and the bias = 0.
  # 4. conv2d thus yields int32 with the scale = s1 * s2, the bias = 0. A caller can apply an activation function or requantize-rescale.

  from broquant.QTensor import quantize_tensor, dequantize_tensor
  input = dequantize_tensor(input)
  weight = dequantize_tensor(weight)
  if bias is not None:
    bias = dequantize_tensor(bias)
  return quantize_tensor(conv2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding))
