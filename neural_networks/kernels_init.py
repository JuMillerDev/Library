import numpy as np

#suitable for Sigmoid and Tanh activations
def xavier_glorot_init(shape):
    if len(shape) == 2:
        size_in, size_out = shape[0], shape[1]
    else:
        size_in = np.prod(shape[1:])

    limit = np.sqrt(1.0 / size_in)
    return np.random.uniform(-limit, limit, size=shape)

#suitable for Sigmoid and Tanh activations
def xavier_glorot_normalized_init(shape):
    if len(shape) == 2:
        size_in, size_out = shape[0], shape[1]
    else:
        size_in = np.prod(shape[1:])
        size_out = shape[0]

    limit = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-limit, limit, size=shape)

#suitable for Relu or Leaky Relu activation
def he_init(shape):
    if(len(shape) == 2):
        input_size, output_size = shape
        size = input_size * output_size #For 1D shape, e.g. Dense layer (input_size,output_size)
    else:
        size = np.prod(shape[:1]) #For multidimensional shape, e.g. Convolutional Layer
        
    scale = np.sqrt(2 / size)
    return np.random.randn(*shape) * scale
        