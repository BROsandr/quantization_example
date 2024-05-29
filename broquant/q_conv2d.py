import torch.nn.functional

def q_conv2d(input, weight, bias=None, stride=1, padding=0):
  # TODO:
  # Algorithm:
  # 1. conv2d = unfold - matmul - fold
  # 2. unfold, fold: keep the pytorch's implementation because these operations are data manipulations.
  # 3. matmul will yield int32 with scale = s1 * s2. Fold will do channel-wise addition of the matmul's result (int32 + int32). Then we add to the fold's result the bias, quantized with the scale = s1 * s2 and the bias = 0.
  # 4. conv2d thus yields int32 with the scale = s1 * s2, the bias = 0. A caller can apply an activation function or requantize-rescale.

  from math import floor

  if isinstance(stride, int): stride = (stride, stride)
  if isinstance(padding, int): padding = (padding, padding)
  dilation = [1, 1]

  inp_unf = torch.nn.functional.unfold(input=input, kernel_size=weight.shape[-2:], padding=padding, stride=stride)

  transpose = lambda tensor: tensor.t() if len(tensor.shape) <= 2 else tensor.transpose(1, 2)

  out_unf = transpose(torch.matmul(transpose(inp_unf), (weight.view(weight.size(0), -1).t())))

  # see h_out, w_out formulas in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
  h_out = floor((input.shape[-2] + 2 * padding[0] - dilation[0] * (weight.shape[-2] - 1) - 1) / stride[0] + 1)
  w_out = floor((input.shape[-1] + 2 * padding[1] - dilation[1] * (weight.shape[-1] - 1) - 1) / stride[1] + 1)
  out = torch.nn.functional.fold(input=out_unf, output_size=(h_out, w_out), kernel_size=(1, 1))
  if bias is not None: out += bias[..., None, None]
  return out
