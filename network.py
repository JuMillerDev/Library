def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward_propagation(output)
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
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward_propagation(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")