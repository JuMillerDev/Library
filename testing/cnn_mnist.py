import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical

from testing.utils import *

from neural_networks.layers.dense_layer import Dense
from neural_networks.layers.dropout_layer import Dropout
from neural_networks.layers.convolutional_layer import Convolutional
from neural_networks.layers.reshape_layer import Reshape
from neural_networks.layers.max_pooling_layer import MaxPooling2D
from neural_networks.functions.activation_functions import *
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


def cnn_learning_rate_test(iterations: int = 2):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)
    x_test, y_test = preprocess_data(x_test, y_test, 1000)

    num_epochs = 11
    tested_learning_rates = [0.01,0.05,0.1,0.2,0.3]
    files = []
    subplot_legend_labels = []
    learning_rate_accuracies = np.zeros(len(tested_learning_rates))

    for index,learning_rate in enumerate(tested_learning_rates):
        print("Learning rate: ",learning_rate)
        testing_epochs_errors = np.zeros(num_epochs)
        for i in range(iterations):
            print("Iteration: {}/{}".format(i+1,iterations))

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
            training_errors = train(
                network,
                categorical_cross_entropy,
                categorical_cross_entropy_prime,
                x_train,
                y_train,
                epochs=num_epochs,
                learning_rate=learning_rate,
                return_training_error=True
            )

            testing_epochs_errors += training_errors
            learning_rate_accuracies[index] += test_network(network,x_test,y_test)

        testing_epochs_errors /= iterations
        learning_rate_accuracies /= iterations

        np.savetxt("cnn_learning_rate_{}.txt".format(learning_rate), testing_epochs_errors, delimiter='\n', fmt='%f')
        files.append("cnn_learning_rate_{}.txt".format(learning_rate))
        subplot_legend_labels.append("wsp. uczenia = {}".format(learning_rate))

    np.savetxt("cnn_learning_rate_accuracies.txt", learning_rate_accuracies, delimiter='\n', fmt='%f')
    plot(files,subplot_legend_labels,'cnn_learning_rate_test_plot','Testy współczynnika uczenia dla CNN','błędy klasyfikacji','iteracja trenowania')

def cnn_layers_test(iterations: int = 2):
    tested_architectures = [
        [
            Convolutional(input_shape=(1,28,28), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Reshape(input_shape=(5,13,13), output_shape=(5*13*13,1)),
            Dense(input_size=(5*13*13), output_size=10),
            Softmax()
        ],
        [
            Convolutional(input_shape=(1,28,28), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Convolutional(input_shape=(5,13,13), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Reshape(input_shape=(5,5,5), output_shape=(5*5*5,1)),
            Dense(input_size=(5*5*5), output_size=10),
            Softmax()
        ],
        [
            Convolutional(input_shape=(1,28,28), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Convolutional(input_shape=(5,13,13), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Convolutional(input_shape=(5,5,5), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            Reshape(input_shape=(5,3,3), output_shape=(5*3*3,1)),
            Dense(input_size=5*3*3,output_size=10),
            Softmax()
        ],
        [
            Convolutional(input_shape=(1,28,28), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Reshape(input_shape=(5,13,13), output_shape=(5*13*13,1)),
            Dense(input_size=(5*13*13), output_size=300),
            Relu(),
            Dense(input_size=300, output_size=10),
            Softmax()
        ],
        [
            Convolutional(input_shape=(1,28,28), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Reshape(input_shape=(5,13,13), output_shape=(5*13*13,1)),
            Dense(input_size=(5*13*13), output_size=400),
            Relu(),
            Dense(input_size=400, output_size=100),
            Relu(),
            Dense(input_size=100, output_size=10),
            Softmax()
        ],
        [
            Convolutional(input_shape=(1,28,28), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Convolutional(input_shape=(5,13,13), kernel_size=3, depth=5, kernels_init='he'),
            Relu(),
            MaxPooling2D(),
            Reshape(input_shape=(5,5,5), output_shape=(5*5*5,1)),
            Dense(input_size=(5*5*5), output_size=84),
            Relu(),
            Dense(input_size=84, output_size=10),
            Softmax()
        ]
    ]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)
    x_test, y_test = preprocess_data(x_test, y_test, 1000)

    num_epochs = 11
    files = []
    subplot_legend_labels = [
        '1 konw. + 1 gęsta', 
        '2 konw. + 1 gęsta', 
        '3 konw. + 1 gęsta', 
        '1 konw. + 2 gęsta',
        '1 konw. + 3 gęsta',
        '2 konw. + 2 gęsta'
    ]
    model_accuracies = np.zeros(len(tested_architectures))

    for index,network in enumerate(tested_architectures):
        print("architecture: {}/{}".format(index+1,len(tested_architectures)))
        testing_epochs_errors = np.zeros(num_epochs)
        for i in range(iterations):
            print("Iteration: {}/{}".format(i+1,iterations))

            training_errors = train(
               network,
               categorical_cross_entropy,
               categorical_cross_entropy_prime,
               x_train,
               y_train,
               epochs=num_epochs,
               learning_rate= 0.2,
               return_training_error=True
            )

            testing_epochs_errors += training_errors
            model_accuracies[index] += test_network(network,x_test,y_test)
        
        testing_epochs_errors /= iterations
        model_accuracies /= iterations

        np.savetxt("cnn_layers_{}.txt".format(subplot_legend_labels[index]), testing_epochs_errors, delimiter='\n', fmt='%f')
        files.append("cnn_layers_{}.txt".format(subplot_legend_labels[index]))
    np.savetxt("cnn_layers_accuracies.txt", model_accuracies, delimiter='\n', fmt='%f')
    plot(files,subplot_legend_labels,'cnn_layers_test_plot','Testy architektury dla CNN','błędy klasyfikacji','iteracja trenowania')
    
def cnn_activation_functions_test(iterations: int = 2):
    tested_activation_func = [
        ('Tanh',Tanh, 'xavier_norm'),
        ('Sigmoid', Sigmoid, 'xavier_norm'),
        ('RELU', Relu, 'he'),
        ('Przeciekający RELU', Leaky_Relu, 'he')
    ]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)
    x_test, y_test = preprocess_data(x_test, y_test, 1000)

    num_epochs = 11
    files = []
    functions_accuracies = np.zeros(len(tested_activation_func))

    for index,activation_function in enumerate(tested_activation_func):
        print("function: {}/{}".format(index+1,len(tested_activation_func)))
        testing_epochs_errors = np.zeros(num_epochs)
        for i in range(iterations):
            print("Iteration: {}/{}".format(i+1,iterations))

            kernels_init = activation_function[2]
            function = activation_function[1]

            network = [
                Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5, kernels_init=kernels_init),
                function(),
                MaxPooling2D(),
                Convolutional(input_shape=(5, 13, 13), kernel_size=3, depth=5, kernels_init=kernels_init),
                function(),
                MaxPooling2D(),
                Reshape(input_shape=(5, 5, 5), output_shape=(5 * 5 * 5,1)),
                Dense(input_size=125, output_size=100, kernels_init=kernels_init),
                function(),
                Dense(input_size=100, output_size=84, kernels_init=kernels_init),
                function(),
                Dense(input_size=84, output_size=10),
                Softmax()
            ]

            training_errors = train(
               network,
               categorical_cross_entropy,
               categorical_cross_entropy_prime,
               x_train,
               y_train,
               epochs=num_epochs,
               learning_rate= 0.2,
               return_training_error=True
            )

            testing_epochs_errors += training_errors
            functions_accuracies[index] += test_network(network,x_test,y_test)

        testing_epochs_errors /= iterations
        functions_accuracies /= iterations

        np.savetxt("cnn_activations_{}.txt".format(activation_function[0]), testing_epochs_errors, delimiter='\n', fmt='%f')
        files.append("cnn_activations_{}.txt".format(activation_function[0]))
    
    subplot_legend_labels = [t[0] for t in tested_activation_func]
    np.savetxt("cnn_activations_accuracies.txt", functions_accuracies, delimiter='\n', fmt='%f')
    plot(files,subplot_legend_labels,'cnn_activations_plot','Testy funkcji aktywacyjnych dla CNN','błędy klasyfikacji','iteracja trenowania')

def cnn_run(iterations: int = 1):
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
    # cnn_learning_rate_test(100)
    # cnn_layers_test(100)
    cnn_activation_functions_test(100)
