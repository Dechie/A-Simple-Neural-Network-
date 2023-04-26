import numpy as np
import matplotlib.pyplot as plt

# sigmoid function to determine the output of a sigle neuron
# based on weight and bias values
def sigmoid(x):
    # example activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))



# derivative of sigmoid function: part of the differential equation
# that determines the training of the network and loss calculation
def deriv_sigmoid(x):
    # derivative of example sigmoid: f'(x) = f(x) (1 - f(x))
    # using the concept of functional programming
    # passing the function as a variable
    fx = sigmoid(x)
    return fx * (1 - fx)


# function to calculate loss value 
# i.e. difference between true value and predicted value
def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    # take the difference of each true and pred value, square it, then calculate mean
    return ((y_true - y_pred) ** 2).mean()
    

# Neuron class (represents single neuron)
class Neuron: 
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # weight inputs, add bias, then use activation function

        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# two arrays to store data from the neural network, 
# we will need them later to plot a graph
time_index = []
losses = []

# neural network class
class OurNeuralNetwork: 
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
        # declare weights
        self.w1 = np.random.normal()    # hidden neuron 1
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()    # hidden neuron 2
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()    # output neuron
        self.w6 = np.random.normal()

        # declare biases 
        # there will be 3 of them: for the 3 neuronsself.
        self.b1 = np.random.normal()    # hidden neuron 1
        self.b2 = np.random.normal()    # hidden neuron 2
        self.b3 = np.random.normal()    # hidden neuron 3


    def feedforward(self, x):
        # feed-forward function for the overall neural network
        # x is an array with 2 elementsself.
        
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1

    def train(self, data, all_y_trues):

        '''
        - data is a (n X 2) numpy array, where n = number of samples in datasetself.
        - all_y_trues is a numpy array with n elements.
        
        Elements in all_y_trues correspond to those in data.
        '''

        learn_rate = 0.1
        epochs = 1000 # number of times to loop thru datasetself
        graph_plot = 0  # later used to plot the x axis of graph
        
        for epoch in range (epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward function

                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * x[0] + self.w6 * x[1] + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1
                
                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                
                # Calculate total loss at the end of each epoch

                if epoch % 10 == 0:
                    time_index.append(epoch)
                    
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    
                    loss = mse_loss(all_y_trues, y_preds)
                    losses.append(loss)
                    
                    print("Epoch %d loss: %.3f" % (epoch, loss))


# Define example dataset
data = np.array([

    [-2, -1],   # Betty
    [25, 6],    # Dani
    [17, 4],    # Abdisa
    [-15, -6]   # Helen
])

all_y_trues = np.array([  # 1 represents female, 0 represents male

    1, # betty 
    0, # dani
    0, # abdisa
    1, # helen
    
])


# finally, train the network

network = OurNeuralNetwork() 
network.train(data, all_y_trues)

# convert the recorded values to np arrays
x_axis = np.array(time_index)
y_axis = np.array(losses)

# plot graph with the np arrays
plt.plot(x_axis, y_axis)
plt.xlabel('time')
plt.ylabel('loss')

plt.title('training neural network')
plt.show()























