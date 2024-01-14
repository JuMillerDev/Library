from neural_networks.kernels_init import he_init, le_cun_init, xavier_glorot_init
from neural_networks.layer import LayerInterface
import numpy as np

#also called fully connected layer
class Dense(LayerInterface):
    
    def __init__(self, input_size, output_size, kernels_init="none"):
        self.bias = np.zeros((output_size,1))
        self.num_examples = 0
        
        match(kernels_init):
            case("none"):
                self.weights = np.random.randn(output_size, input_size)
            case("he"):
                self.weights = he_init((output_size, input_size))
            case("xavier"):
                self.weights = xavier_glorot_init((input_size, output_size))
            case("lecun"):
                self.weights = le_cun_init(input_size, output_size)
            case _:
                print("there is no such kernel initialization")
                self.weights = np.random.randn(input_size, output_size)
        
    def forward_propagation(self, input):
        self.input = input
        #y = w * x + b where w, x are matrices 
        # print("w: ", self.weights.shape, " i: ", self.input.shape)
        return np.dot(self.weights, self.input) + self.bias
    
    def backward_propagation(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T) / self.input.shape[0]
        #test
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True) / self.input.shape[0]
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * output_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient
