# A-Simple-Neural-Network-
simple neural network built from scratch
A simple neural network built from scrach

simple intro This project is a practice project to build a neural network from scratch. this is an example neural network.

the basic idea of neural network is; to model a group of neurons that accept data, process it and pass it onto other neurons.

the basic idea in the simplest form, a neuron takes two input values, and produces one output value.

the neuron multiplies each neuron by the weight values using dot product, then adds the bias value. and after this, the result is passed into the activation function, and the result would be the output of that given neuron.

this project has 3 files, one file showcases a the implementation of a single neuron. the second file, shows a simple neural network with two neurons in the hidden layer that take the input, and the output neuron.

training the neural network: once we build a neural network, we have to make sure the weight is minimized. that is done by first calculating the loss of the network, i.e., the difference between the true value and the value predicted by the neural network. and this is done with the mean squared error.

after that, we use some more mathematics to determine how we can minimize the loss by changing the value of one of the inputs (either the inputs of the hidden layer neuron, or, the input of the output layer, i.e., the outputs of the hidden layer neurons.)

we apply partial derivatives to determine this, and once we have plugged the numbers, we can insert random values as the weights to the neuron inputs, and check if the predicted outputs are closer or farther from the true values.

the third file, "neuralNetwork2.py", contains the neuron implementation, a neuralNetwork with weights and inputs varying (unlike in the second file), as well as the MSE calculation and the derivative calculation functions.

and finally, the last file contains a graph, which can help visualize the final output of the system.

![image](https://user-images.githubusercontent.com/104849949/234575719-b67f666a-163b-437a-ac6e-3cf5b6a11b54.png)

this is what the final graph looks like.
