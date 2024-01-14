from neural_networks.layers.dropout_layer import Dropout

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

#todo end optimizers and carry them to separate file
def sdg_optimizer(network, learning_rate):
    for layer in network:
        if(hasattr(layer, 'weights')):
            pass
        if(hasattr(layer, 'kernels')):
            pass
        if(hasattr(layer, 'biases')):
            pass
            
def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
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