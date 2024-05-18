from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path
from utils import test, train


class Model(nn.Module):
    def __init__(self, mnist=True):

        super(Model, self).__init__()
        if mnist:
          num_channels = 1
        else:
          num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x: torch.Tensor)->torch.Tensor:

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @classmethod
    def create(cls, load_model=True, model_path=Path('./data/mnist_cnn.pt'), save_model=True):

        batch_size = 64
        test_batch_size = 64
        epochs = 3
        lr = 0.01
        momentum = 0.5
        seed = 1
        log_interval = 500
        no_cuda = False

        use_cuda = not no_cuda and torch.cuda.is_available()

        torch.manual_seed(seed)

        device = torch.device("cuda" if use_cuda else "cpu")
        model = cls().to(device)

        if not load_model:

          kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
          train_loader = torch.utils.data.DataLoader(
              datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
              batch_size=batch_size, shuffle=True, **kwargs)

          test_loader = torch.utils.data.DataLoader(
              datasets.MNIST('../data', train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
              batch_size=test_batch_size, shuffle=True, **kwargs)


          optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
          args = {}
          args["log_interval"] = log_interval
          for epoch in range(1, epochs + 1):
              train(args, model, device, train_loader, optimizer, epoch)
              test(args, model, device, test_loader)

          if (save_model):
              torch.save(model.state_dict(), model_path)
        else:
          state_dict = torch.load(model_path)
          model.load_state_dict(state_dict)

        return model
