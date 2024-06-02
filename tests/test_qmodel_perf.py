import sys
if __name__ == '__main__': sys.path.append('.')

from broquant.const import MNIST_MODEL_PATH, MNIST_DATASET_PATH
from broquant.Model import Model
import torch
from torchvision import datasets, transforms
import torch.utils.data
from broquant.QModel import gatherStats, QModel

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

dataset_path = MNIST_DATASET_PATH
model = Model.create(load_model=True, model_path=MNIST_MODEL_PATH, dataset_path=dataset_path)

kwargs = {'num_workers': 1, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST(str(dataset_path), train=False, transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
  batch_size=1, shuffle=True, **kwargs)
stats = gatherStats(model, test_loader)
q_model = QModel(model=model, stats=stats)

data, _ = next(iter(test_loader))

logger.debug(f'stats: {stats}')

output = q_model(data)

logger.debug(f'output: {output}')
