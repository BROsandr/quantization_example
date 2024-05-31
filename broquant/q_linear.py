def q_linear(input, weight, bias=None):
  """
    Generic implementation of torch.nn.functional.linear.
  """

  res = input @ weight.t()
  if not (bias is None):
    res += bias
  return res
