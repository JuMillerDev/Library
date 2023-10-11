import numpy as np

#especially popular for Sigmoid and Tanh activations
def xavier_glorot_init(input_size, output_size):
    return np.random.randn(output_size, input_size) * np.sqrt(2 / (input_size, output_size))

#suitable for Relu or Leaky Relu activation
def he_init(shape):
    if(len(shape) == 2):
        size = shape #For 1D shape, e.g. Dense layer (input_size,output_size)
    else:
        size = np.prod(shape[:1]) #For multidimensional shape, e.g. Convolutional Layer
        
    scale = np.sqrt(2 / size)
    return np.random.randn(*shape) * scale
        

#suitable for Tanh activation
def le_cun_init(input_size, output_size):
    return np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)