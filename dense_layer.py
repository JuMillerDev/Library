from layer import LayerInterface
import numpy as np

#also called fully connected layer
class Dense(LayerInterface):
    
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros((output_size,1))
        
    def forward_propagation(self, input):
        self.input = input
        #y = w * x + b where w, x are matrices 
        return np.dot(self.weights, self.input) + self.bias
    
    def backward_propagation(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        #test
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * output_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient
