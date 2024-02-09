import copy
import os
import sys
import numpy as np
import cv2
from keras.datasets import mnist
from keras.utils import to_categorical

from neural_networks.layers.dense_layer import Dense
from neural_networks.layers.convolutional_layer import Convolutional
from neural_networks.layers.reshape_layer import Reshape
from neural_networks.functions.activation_functions import *
from neural_networks.functions.loss_functions import binary_cross_entropy, binary_cross_entropy_prime, categorical_cross_entropy, categorical_cross_entropy_prime
from neural_networks.network import train, predict
from testing.utils import *

def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

def preprocess_user_data(pictures_location:str, labels_file_location:str, show_preprocessed_image: bool = False):
    if not os.path.isdir(pictures_location):
        print(f"Error: {pictures_location} is not a valid directory.")
        return
    
    if not os.path.isfile(labels_file_location) or not os.path.splitext(labels_file_location)[1].lower() == '.txt':
        print(f"Error: {labels_file_location} is not a valid txt file.")
        return
    
    # get pictures from folder
    picture_location_files = os.listdir(pictures_location)

    # Filter out only picture files (you can extend this list for other formats)
    picture_extensions = {".jpg", ".jpeg", ".png", ".gif"}
    joined_paths = [os.path.join(pictures_location, file) for file in picture_location_files]
    picture_files = [file for file in joined_paths if os.path.isfile(file) and os.path.splitext(file)[1].lower() in picture_extensions]

    processed_images = []
    
    for image_path in picture_files:
        # Step 1: Resize to 28x28

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (28,28), interpolation = cv2.INTER_LINEAR)
        img_resized = cv2.bitwise_not(img_resized)

        if show_preprocessed_image:

            # Display side by side
            plt.imshow(img_resized, cmap='gray')
            plt.title("Preprocessed image")

            plt.show()

        img_array = np.array(img_resized) / 255.0

        img_reshaped = img_array.reshape((1, 28, 28))

        processed_images.append(img_reshaped)

    # Stack the processed images vertically
    images_array = np.vstack(processed_images)
    images_array = images_array.reshape(images_array.shape[0], 28 * 28, 1)

    with open(labels_file_location, 'r') as labels_file:
        labels = [int(line.strip()) for line in labels_file]

    labels = to_categorical(labels, num_classes=10)
    labels = labels.reshape(labels.shape[0], 10, 1)

    return (images_array,labels)

def test_network(network, x_test, y_test):
        # test (MNIST test data)
    correct_predictions = 0
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        if(np.argmax(output) == np.argmax(y)):
            correct_predictions += 1
    return correct_predictions/len(x_test)

def mlp_learning_rate_test(iterations: int = 2, user_test_data: tuple = None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)

    if user_test_data == None:
        x_test, y_test = preprocess_data(x_test, y_test, 1000)
    else:
        x_test, y_test = user_test_data[0], user_test_data[1]

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
                Dense(input_size=784, output_size=512, kernels_init="he"),
                Relu(),
                Dense(input_size=512, output_size=100, kernels_init="he"),
                Relu(),
                Dense(input_size=100, output_size=10),
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
        learning_rate_accuracies[index] /= iterations

        print('average accuracy: ',learning_rate_accuracies[index])

        np.savetxt("mlp_learning_rate_{}.txt".format(learning_rate), testing_epochs_errors, delimiter='\n', fmt='%f')
        files.append("mlp_learning_rate_{}.txt".format(learning_rate))
        subplot_legend_labels.append("wsp. uczenia = {}".format(learning_rate))

    np.savetxt("mlp_learning_rate_accuracies.txt", learning_rate_accuracies, delimiter='\n', fmt='%f')
    plot(files,subplot_legend_labels,'mlp_learning_rate_test_plot','Testy współczynnika uczenia dla MLP','błędy klasyfikacji','iteracja trenowania')


def mlp_layers_test(iterations: int = 2, user_test_data: tuple = None):
    tested_architectures = [
        [
            Dense(input_size=784, output_size=10),
            Softmax()
        ],
        [
            Dense(input_size=784, output_size=400, kernels_init='he'),
            Relu(),
            Dense(input_size=400, output_size=10),
            Softmax()
        ],
        [
            Dense(input_size=784, output_size=400, kernels_init='he'),
            Relu(),
            Dense(input_size=400, output_size=100, kernels_init='he'),
            Relu(),
            Dense(input_size=100, output_size=10),
            Softmax()
        ],
        [
            Dense(input_size=784, output_size=500, kernels_init='he'),
            Relu(),
            Dense(input_size=500, output_size=300, kernels_init='he'),
            Relu(),
            Dense(input_size=300, output_size=80, kernels_init='he'),
            Relu(),
            Dense(input_size=80, output_size=10),
            Softmax()
        ]
    ]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)

    if user_test_data == None:
        x_test, y_test = preprocess_data(x_test, y_test, 1000)
    else:
        x_test, y_test = user_test_data[0], user_test_data[1]

    num_epochs = 11
    files = []
    subplot_legend_labels = [
        'bez warstw ukrytych', 
        '1 warstwa ukryta', 
        '2 warstwy ukryte', 
        '3 warstwy ukryte',
    ]

    model_accuracies = np.zeros(len(tested_architectures))
    for index,network in enumerate(tested_architectures):
        print("architecture: {}/{}".format(index+1,len(tested_architectures)))
        testing_epochs_errors = np.zeros(num_epochs)
        for i in range(iterations):
            print("Iteration: {}/{}".format(i+1,iterations))

            net = copy.deepcopy(network)

            training_errors = train(
               net,
               categorical_cross_entropy,
               categorical_cross_entropy_prime,
               x_train,
               y_train,
               epochs=num_epochs,
               learning_rate= 0.2,
               return_training_error=True
            )

            testing_epochs_errors += training_errors
            model_accuracies[index] += test_network(net,x_test,y_test)
        
        testing_epochs_errors /= iterations
        model_accuracies[index] /= iterations

        print('average accuracy: ',model_accuracies[index])

        np.savetxt("mlp_layers_{}.txt".format(subplot_legend_labels[index]), testing_epochs_errors, delimiter='\n', fmt='%f')
        files.append("mlp_layers_{}.txt".format(subplot_legend_labels[index]))
    np.savetxt("mlp_layers_accuracies.txt", model_accuracies, delimiter='\n', fmt='%f')
    plot(files,subplot_legend_labels,'mlp_layers_test_plot','Testy architektury dla MLP','błędy klasyfikacji','iteracja trenowania')
    
def mlp_activation_functions_test(iterations: int = 2, user_test_data: tuple = None):
    tested_activation_func = [
        ('Tanh',Tanh, 'xavier_norm'),
        ('Sigmoid', Sigmoid, 'xavier_norm'),
        ('RELU', Relu, 'he'),
        ('Przeciekający RELU', Leaky_Relu, 'he')
    ]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)

    if user_test_data == None:
        x_test, y_test = preprocess_data(x_test, y_test, 1000)
    else:
        x_test, y_test = user_test_data[0], user_test_data[1]

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
                Dense(input_size=784, output_size=512, kernels_init=kernels_init),
                function(),
                Dense(input_size=512, output_size=100, kernels_init=kernels_init),
                function(),
                Dense(input_size=100, output_size=10),
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
        functions_accuracies[index] /= iterations

        print('average accuracy: ',functions_accuracies[index])

        np.savetxt("mlp_activations_{}.txt".format(activation_function[0]), testing_epochs_errors, delimiter='\n', fmt='%f')
        files.append("mlp_activations_{}.txt".format(activation_function[0]))
    
    subplot_legend_labels = [t[0] for t in tested_activation_func]
    np.savetxt("mlp_activations_accuracies.txt", functions_accuracies, delimiter='\n', fmt='%f')
    plot(files,subplot_legend_labels,'mlp_activations_plot','Testy funkcji aktywacyjnych dla MLP','błędy klasyfikacji','iteracja trenowania')

def mlp_run(user_test_data: tuple = None):
    # load MNIST from server, if the training is too long limit the amount of training examples, since we're not training on GPU
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)
    if user_test_data == None:
        x_test, y_test = preprocess_data(x_test, y_test, 1000)
    else:
        x_test, y_test = user_test_data[0], user_test_data[1]

    # Classic LeNet-5 CNN implementation
    network = [
            Dense(input_size=784, output_size=400, kernels_init='he'),
            Relu(),
            Dense(input_size=400, output_size=10),
            Softmax()
        ]

    # train
    train(
        network,
        categorical_cross_entropy,
        categorical_cross_entropy_prime,
        x_train,
        y_train,
        epochs=14,
        learning_rate=0.3
    )

    accuracy = test_network(network,x_test,y_test)
    print("accuracy rate: ", accuracy)

    return network

if __name__ == "__main__":
    num_of_args = len(sys.argv)
    
    if num_of_args == 1:
        mlp_learning_rate_test(1)
        mlp_activation_functions_test(1)
        mlp_layers_test()
    elif num_of_args == 3:
        user_data = preprocess_user_data(sys.argv[1], sys.argv[2]) 
        mlp_learning_rate_test(1, user_data)
        mlp_activation_functions_test(1, user_data)
        mlp_layers_test(1, user_data)
    else:
        print('Wrong amount of arguments: \n for user data tests enter: py -m testing.mlp_mnist <folder with pictures> <file with class labels> \n for MNIST data tests enter: py -m testing.mlp_mnist')