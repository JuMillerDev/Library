import numpy as np
from neural_networks.layer import LayerInterface


class Dropout(LayerInterface):
    def __init__(self, keep_probability):
        self.keep_probability = keep_probability
        self.mask = None

    def forward_propagation(self, input, training: bool):
        if training:
            self.mask = (np.random.rand(*input.shape) < self.keep_probability)
            return self._apply_mask(input, self.mask)
        else:
            return input
    
    def backward_propagation(self, output_gradient, learning_rate):
        return self._apply_mask(output_gradient, self.mask)
    
    def _apply_mask(self, input, mask: np.array):
        input *= mask
        input /= self.keep_probability
        return input