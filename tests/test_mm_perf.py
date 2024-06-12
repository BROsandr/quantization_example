import random
import torch
from broquant.QTensor import QTensor
import timeit

iteration_time = 0

timeit_repetitions = 10

iterations = 1

def time_random(time_repetitions):
  h1_dim = random.randint(576, 576)
  w1_dim = random.randint(25, 25)
  w2_dim = random.randint(20, 20)
  a = torch.rand(h1_dim, w1_dim, requires_grad=False)
  b = torch.rand(w1_dim, w2_dim, requires_grad=False)
  with torch.no_grad():
    return (timeit.timeit(lambda: QTensor.quantize(a).mm(QTensor.quantize(b)), number=time_repetitions) / timeit_repetitions)

for i in range(iterations):
  iteration_time += time_random(timeit_repetitions)
  print(f'iteration: {i+1}/{iterations}')

print(f'Took: {iteration_time / iterations}')
