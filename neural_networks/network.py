from neural_networks.layers.dropout_layer import Dropout
import numpy as np

def predict(network, input, training: bool = False):
    output = input
    for layer in network:
        if isinstance(layer, Dropout):
            # use dropout only for training, by passing training boolean
            output = layer.forward_propagation(output, training)
        else:
            output = layer.forward_propagation(output)
    # print(output.shape)
    return output

            
def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True, return_training_error = False):
    if return_training_error:
        error_in_epoch = np.zeros(epochs)

    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x, True)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                # print(layer)
                grad = layer.backward_propagation(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
        if return_training_error:
            error_in_epoch[e] = error
            
    if return_training_error:
        return error_in_epoch