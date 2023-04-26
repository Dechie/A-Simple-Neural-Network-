import numpy as np

def sigmoid(x):
    # example activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # weight inputs, add bias, then use activation function

        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3]) # x1 = 2, x2 = 3  

print(n.feedforward(x)) 
