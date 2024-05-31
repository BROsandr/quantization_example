import torch
from typing import Iterable

def q_matmul(input: torch.Tensor, other: torch.Tensor)->torch.Tensor:
  """
    Generic partial implementation of torch.matmul. Utilizes torch.mm.
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
