import numpy as np

def sigmoid(x):
    # example activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:  # single neuron
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # weight inputs, add bias, then use activation function

        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNetwork: # network of neurons with layers
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (n1, n2)
        - an output layer with 1 neuron (o1)

    Each neuron has the same weights and bias 
    (for the sake of simplicity)

    w = [0, 1]
    b = 0
    
    '''

    def __init__(self): #constructor
        weights = np.array([0,1])
        bias = 0

        # declare Neuron objects 
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)


    def feedforward(self, x):
        # feed-forward function for the overall neural network

        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # the inputs for o1 are the outputs from h1 and h2, defined above
        # the input is an array, so it takes out_h1 and out_h2 as array
        
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # answer should be ~ 0.7216
    














    




















