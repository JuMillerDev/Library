import numpy as np
from neural_networks.layer import LayerInterface

class MaxPooling2D(LayerInterface):
    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size
        self.input = None
        self.mask = None

    def forward_propagation(self, input):
        # if the picture is grayscale
        if len(input.shape) == 3:
            input = np.expand_dims(input, axis=-1)

        self.input = input
        batch_size, input_height, input_width, num_channels = input.shape

        output_height = input_height // self.pool_size[0]
        output_width = input_width // self.pool_size[1]

        output = np.zeros((batch_size, output_height, output_width, num_channels))

        self.mask = np.zeros_like(input)

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_row = i * self.pool_size[0]
                        end_row = start_row + self.pool_size[0]
                        start_col = j * self.pool_size[1]
                        end_col = start_col + self.pool_size[1]

                        pool_region = input[b, start_row:end_row, start_col:end_col, c]
                        max_value = np.max(pool_region)

                        output[b, i, j, c] = max_value

                        # Save the mask to use during backpropagation
                        mask = (pool_region == max_value)
                        self.mask[b, start_row:end_row, start_col:end_col, c] = mask

        return output

    def backward_propagation(self, output_gradient, learning_rate):

        if len(output_gradient.shape) == 3:
            batch_size, output_height, output_width = output_gradient.shape
            num_channels = 1  # Assuming a single channel in this case (grayscale)
        else:
            batch_size, output_height, output_width, num_channels = output_gradient.shape

        input_gradient = np.zeros_like(self.input)

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_row = i * self.pool_size[0]
                        end_row = start_row + self.pool_size[0]
                        start_col = j * self.pool_size[1]
                        end_col = start_col + self.pool_size[1]

                        if len(output_gradient.shape) == 3:
                            output_grad = output_gradient[b, i, j]
                        else:
                            output_grad = output_gradient[b, i, j, c]

                        mask = self.mask[b, start_row:end_row, start_col:end_col, c]

                        # Distribute the gradient to the locations of the max values
                        input_gradient[b, start_row:end_row, start_col:end_col, c] += mask * output_grad

        # If there's a single channel, remove the last dimension
        if num_channels == 1:
            input_gradient = np.squeeze(input_gradient, axis=-1)

        return input_gradient
