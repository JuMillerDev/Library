import numpy as np
from layer import LayerInterface


class Activation(LayerInterface):
    def __init__(self,activation_function,activation_prime):
        self.activation_function = activation_function
        self.activation_prime = activation_prime
    
    def forward_propagation(self, input):
        self.input = input
        return self.activation_function(self.input)
    
    def backward_propagation(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))