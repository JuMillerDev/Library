import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical

from testing.utils import load_model, save_model

# To turn off tensorflow information when running code
keras.utils.disable_interactive_logging()

from neural_networks.layers.dense_layer import Dense
from neural_networks.layers.dropout_layer import Dropout
from neural_networks.layers.convolutional_layer import Convolutional
from neural_networks.layers.reshape_layer import Reshape
from neural_networks.layers.max_pooling_layer import MaxPooling2D
from neural_networks.functions.activation_functions import Leaky_Relu, Relu, Sigmoid, Softmax
from neural_networks.functions.loss_functions import binary_cross_entropy, binary_cross_entropy_prime, categorical_cross_entropy, categorical_cross_entropy_prime
from neural_networks.network import train, predict

def preprocess_data(x, y, limit):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x[:limit], y[:limit]

def test_network(network, x_test, y_test):
        # test (MNIST test data)
    correct_predictions = 0
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        if(np.argmax(output) == np.argmax(y)):
            correct_predictions += 1
    return correct_predictions/len(x_test)


def cnn_for_comparison(iterations: int = 1):
    # load MNIST from server, if the training is too long limit the amount of training examples, since we're not training on GPU
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)
    x_test, y_test = preprocess_data(x_test, y_test, 1000)

    if iterations > 1: file = open("cnn_comparison", "a")

    for i in range(iterations):
        # Classic LeNet-5 CNN implementation
        network = [
            Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5, kernels_init="he"),
            Relu(),
            MaxPooling2D(),
            Convolutional(input_shape=(5, 13, 13), kernel_size=3, depth=5, kernels_init="he"),
            Relu(),
            MaxPooling2D(),
            Reshape(input_shape=(5, 5, 5), output_shape=(5 * 5 * 5,1)),
            Dense(input_size=125, output_size=100, kernels_init="he"),
            Relu(),
            Dense(input_size=100, output_size=84, kernels_init="he"),
            Relu(),
            Dense(input_size=84, output_size=10),
            Softmax()
        ]

        # train
        train(
            network,
            categorical_cross_entropy,
            categorical_cross_entropy_prime,
            x_train,
            y_train,
            epochs=5,
            learning_rate=0.5
        )

        accuracy = test_network(network,x_test,y_test)
        print("accuracy rate: ", accuracy)

        if(iterations > 1): file.write(i, " ," ,accuracy)
    return network

if __name__ == "__main__":
    n = cnn_for_comparison()
    save_model(n)
    m = load_model()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test, y_test = preprocess_data(x_test, y_test, 1000)
    print("testing new")
    acc = test_network(m,x_test,y_test)
    print(acc)
