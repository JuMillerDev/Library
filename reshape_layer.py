import numpy as np
from layer import LayerInterface

class Reshape(LayerInterface):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propagation(self, input):
        return np.reshape(input, self.output_shape)

    def backward_propagation(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)