import numpy as np
from nn import NN
from nn import Relu, Linear, SquaredLoss, CELoss
from utils import data_loader, acc, save_plot, loadMNIST, onehot

x_train, label_train = loadMNIST('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
x_test, label_test = loadMNIST('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
y_train = onehot(label_train)
y_test = onehot(label_test)

model = NN(Relu(), SquaredLoss(), hidden_layers=[128, 128], input_d=784, output_d=10)
model.print_model()
training_data, dev_data = {"X":x_train, "Y":y_train}, {"X":x_test, "Y":y_test}
from run_nn import train_1pass
model, plot_dict = train_1pass(model, training_data, dev_data, learning_rate=1e-2, batch_size=64)