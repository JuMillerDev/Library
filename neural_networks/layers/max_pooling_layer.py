import numpy as np
from neural_networks.layer import LayerInterface

class MaxPooling2D(LayerInterface):
    def __init__(self, pool_size: tuple[int,int]=(2, 2), stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.shape = None
        self.mask = {}

    def forward_propagation(self, input):

        self.shape = np.array(input, copy=True)
        batch_size, input_height, input_width = input.shape #simplified version for grayscale images

        pool_height,pool_width = self.pool_size
        output_height = 1 + (input_height - pool_height) // self.stride
        output_width = 1 + (input_width - pool_width) // self.stride
        output = np.zeros((batch_size,output_height,output_width))

        for i in range(output_height):
            for j in range(output_width):
                start_height = i * self.stride
                end_height = start_height + pool_height
                start_width = j * self.stride
                end_width = start_width + pool_width
                input_slice = input[:,start_height:end_height,start_width:end_width]
                self.save_mask(input=input_slice, cords=(i,j))
                output[:,i,j] = np.max(input_slice, axis=(1,2))

        return output

    def backward_propagation(self, output_gradient, learning_rate):
        output = np.zeros_like(self.shape)
        _, output_height, output_width = output_gradient.shape
        pool_height,pool_width = self.pool_size

        for i in range(output_height):
            for j in range(output_width):
                start_height = i * self.stride
                end_height = start_height + pool_height
                start_width = j * self.stride
                end_width = start_width + pool_width
                output[:, start_height:end_height, start_width:end_width] += ( 
                    output_gradient[:,i:i+1,j:j+1] * self.mask[(i,j)]
                )

        return output
        
    def save_mask(self, input: np.array, cords: tuple[int,int]):
        mask = np.zeros_like(input)
        n,h,w = input.shape
        input = input.reshape(n,h*w,1)
        index = np.argmax(input, axis=1)

        n_index,c_index = np.indices((n,1))
        mask.reshape(n,h*w,1)[n_index,index,c_index] = 1
        self.mask[cords] = mask