
import numpy as np
from activation_layer import Activation
from layer import LayerInterface


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        #pochodna
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)
        
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

#using alpha to avoid nan when value is <= 0
class Leaky_Relu(Activation):
    def __init__(self, alpha=0.01):
        leaky_relu = lambda x: np.where(x>0, x, x*alpha)
        leaky_relu_prime = lambda x: np.where(x>=0, 1, alpha)
        super().__init__(leaky_relu, leaky_relu_prime)

class Softmax(LayerInterface):
    def __init__(self):
        pass
    
    def forward_propagation(self, input):
        # Shift the input values to avoid numerical instability
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp, axis=0)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)