import numpy as np
from scipy import signal
from neural_networks.layer import LayerInterface
from neural_networks.kernels_init import he_init, xavier_glorot_init

class Convolutional(LayerInterface):
    def __init__(self, input_shape, kernel_size, depth, kernels_init="none"):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)   
        self.biases = np.zeros(self.output_shape)
        
        match(kernels_init):
            case("none"):
                self.kernels = np.random.randn(*self.kernels_shape)
            case("he"):
                self.kernels = he_init(self.kernels_shape)
            case("xavier"):
                self.kernels = xavier_glorot_init(self.kernels_shape)
            case _:
                print("there is no such kernel initialization")
                self.kernels = np.random.randn(*self.kernels_shape)
                

    def forward_propagation(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient