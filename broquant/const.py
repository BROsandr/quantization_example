from importlib.resources import files

DATA_ROOT_PATH = files('broquant.data')
MNIST_DATASET_PATH = DATA_ROOT_PATH / 'mnist'
MNIST_MODEL_PATH = DATA_ROOT_PATH / 'mnist_cnn.pt'
