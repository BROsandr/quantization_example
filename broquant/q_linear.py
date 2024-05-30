def q_linear(input, weight, bias=None):
  res = input @ weight.t()
  if not (bias is None):
    res += bias
  return res
