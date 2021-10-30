import numpy as np
import matplotlib as plt

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = layer_sizes
        self.weights = np.array([ np.random.normal(0, 1, size=(y, x)) for x, y in zip(layer_sizes[:-1], layer_sizes[1:]) ])
        self.biases = np.array([ np.random.normal(0, 1, size=(x, 1)) for x in layer_sizes ] )
        self.activation = self.sigmoid

    def feedforward(self, activation):
        for weight, bias in zip(self.weights, self.biases):
            activation = self.activation(weight*activation + bias)

        return activation

    @staticmethod
    def sigmoid(z):
        return 1/(1+ np.exp(-z))

    @staticmethod
    def ReLU(z):
        return np.max([0, z])

net = NeuralNetwork([3, 5, 1])
print(net.weights)
print(net.biases)