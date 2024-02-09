import copy
import sys
import time
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from testing.cnn_mnist import preprocess_user_data as cnn_user
from testing.mlp_mnist import preprocess_user_data as mlp_user
from testing.knn_mnist import preprocess_user_data as knn_user
from sklearn.metrics import accuracy_score
from machine_learning_algorithms.decision_forest.forest import Forest
from machine_learning_algorithms.knn.knn_algorithm import KNN
from neural_networks.functions.activation_functions import Leaky_Relu, Relu, Softmax
from neural_networks.functions.loss_functions import categorical_cross_entropy, categorical_cross_entropy_prime

from neural_networks.layers.convolutional_layer import Convolutional
from neural_networks.layers.dense_layer import Dense
from neural_networks.layers.max_pooling_layer import MaxPooling2D
from neural_networks.layers.reshape_layer import Reshape
from neural_networks.network import predict, train

def preprocess_data_cnn(x, y, limit, is_own_data:bool, picture_paths:str, label_file_path:str):
    if not is_own_data:
        x = x.reshape(len(x), 1, 28, 28)
        x = x.astype("float32") / 255
        y = to_categorical(y)
        y = y.reshape(len(y), 10, 1)
        return x[:limit], y[:limit]
    else:
        x_y = cnn_user(picture_paths,label_file_path)
        return x_y[0], x_y[1]

def preprocess_data_mlp(x, y, limit,  is_own_data:bool, picture_paths:str, label_file_path:str):
    if not is_own_data:
        x = x.reshape(x.shape[0], 28 * 28, 1)
        x = x.astype("float32") / 255
        y = to_categorical(y)
        y = y.reshape(y.shape[0], 10, 1)
        return x[:limit], y[:limit]
    else:
        x_y = mlp_user(picture_paths,label_file_path)
        return x_y[0], x_y[1]

def preprocess_data_ml(x, y, limit, is_own_data:bool, picture_paths:str, label_file_path:str):
    if not is_own_data:         
        x = x.reshape(x.shape[0], -1)
        x = x.astype("float32") / 255
        return (x[:limit]), (y[:limit])
    else:
        x_y = knn_user(picture_paths,label_file_path)
        return x_y[0], x_y[1]

def test_network(network, x_test, y_test):
        # test (MNIST test data)
    correct_predictions = 0
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        if(np.argmax(output) == np.argmax(y)):
            correct_predictions += 1
    return correct_predictions/len(x_test)
            

def compare_algorithms_accuracy_time(iterations: int = 2, is_own_data: bool = False, picture_paths: str = None, labels_file_path: str = None):

    neural_networks_for_comparison = [
        (preprocess_data_cnn, 
        [
            Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5, kernels_init="he"),
            Leaky_Relu(),
            MaxPooling2D(),
            Convolutional(input_shape=(5, 13, 13), kernel_size=3, depth=5, kernels_init="he"),
            Leaky_Relu(),
            MaxPooling2D(),
            Reshape(input_shape=(5, 5, 5), output_shape=(5 * 5 * 5,1)),
            Dense(input_size=125, output_size=100, kernels_init="he"),
            Leaky_Relu(),
            Dense(input_size=100, output_size=84, kernels_init="he"),
            Leaky_Relu(),
            Dense(input_size=84, output_size=10),
            Softmax()
        ], 0.2, 18),
        (preprocess_data_mlp,
        [
            Dense(input_size=784, output_size=400, kernels_init='he'),
            Relu(),
            Dense(input_size=400, output_size=10),
            Softmax()
        ], 0.3, 13)
    ]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    training_time = np.zeros(len(neural_networks_for_comparison)+2)
    prediction_time = np.zeros(len(neural_networks_for_comparison)+2)
    model_accuracies = np.zeros(len(neural_networks_for_comparison)+2)

    # first process neural networks
    for index,network in enumerate(neural_networks_for_comparison):
        x_train_local, y_train_local = network[0](x_train, y_train, 10000, False, picture_paths, labels_file_path)
        x_test_local, y_test_local = network[0](x_test, y_test, 1000, is_own_data, picture_paths, labels_file_path)
        print('neural network {}/{}'.format(index+1,len(neural_networks_for_comparison)))
        for i in range(iterations):
            print('iteration: {}/{}'.format(i+1,iterations))

            net = copy.deepcopy(network[1])

            start_training_time = time.time()
            train(
                net,
                categorical_cross_entropy,
                categorical_cross_entropy_prime,
                x_train_local,
                y_train_local,
                epochs=network[3],
                learning_rate=network[2],
                return_training_error=True
            )
            end_training_time = time.time()

            start_prediction_time = time.time()
            accuracy = test_network(net, x_test_local, y_test_local)
            end_prediction_time = time.time()

            model_accuracies[index] += accuracy
            training_time[index] += (end_training_time - start_training_time)
            prediction_time[index] += (end_prediction_time - start_prediction_time)
            
        model_accuracies[index] /= iterations
        training_time[index] /= iterations
        prediction_time[index] /= iterations

        print('average accuracy: ',model_accuracies[index])
        print('average training time: ', training_time[index])
        print('average prediction time: ', prediction_time[index])

        np.savetxt('compare_{}_nn.txt'.format(index), [model_accuracies[index],training_time[index],prediction_time[index]], delimiter='\n', fmt='%f')


    # # process knn
    print('knn')
    x_train_local, y_train_local = preprocess_data_ml(x_train, y_train, 10000, False, picture_paths, labels_file_path)
    x_test_local, y_test_local = preprocess_data_ml(x_test, y_test, 10000, is_own_data, picture_paths, labels_file_path)
    for i in range(iterations):
        print("Iteration: {}/{}".format(i+1,iterations))

        start_training_time = time.time()
        model = KNN(K=7, distance_measure='cosine')
        model.fit(x_train_local, y_train_local)
        end_training_time = time.time()

        start_prediction_time = time.time()
        pred = model.predict(x_test_local)
        end_prediction_time = time.time()

        acc = accuracy_score(y_test_local, pred)
        print(acc)
        model_accuracies[len(neural_networks_for_comparison)] += acc
        training_time[len(neural_networks_for_comparison)] += (end_training_time - start_training_time)
        prediction_time[len(neural_networks_for_comparison)] += (end_prediction_time - start_prediction_time)

    model_accuracies[len(neural_networks_for_comparison)] /= iterations
    training_time[len(neural_networks_for_comparison)] /= iterations
    prediction_time[len(neural_networks_for_comparison)] /= iterations

    print('average accuracy: ',model_accuracies[len(neural_networks_for_comparison)])
    print('average training time: ', training_time[len(neural_networks_for_comparison)])
    print('average prediction time: ', prediction_time[len(neural_networks_for_comparison)])

    np.savetxt('compare_{}_ml.txt'.format(len(neural_networks_for_comparison)), [model_accuracies[len(neural_networks_for_comparison)],training_time[len(neural_networks_for_comparison)],prediction_time[len(neural_networks_for_comparison)]], delimiter='\n', fmt='%f')

    y_train_local, y_test_local = np.atleast_2d(y_train_local).T, np.atleast_2d(y_test_local).T
    # process decision forest
    print('df')
    for i in range(iterations):
        print("Iteration: {}/{}".format(i+1,iterations))

        start_training_time = time.time()
        model = Forest(data=x_train_local,
        labels=y_train_local,
        n_trees=100,
        max_features=50,
        bootstrap_features=True,
        max_depth=7,
        min_leaf_points=2)
        end_training_time = time.time()

        start_prediction_time = time.time()
        pred = model.predict(x_test_local)
        end_prediction_time = time.time()

        acc = accuracy_score(y_test_local,pred)
        print(acc)
        model_accuracies[len(neural_networks_for_comparison)+1] += acc
        training_time[len(neural_networks_for_comparison)+1] += (end_training_time - start_training_time)
        prediction_time[len(neural_networks_for_comparison)+1] += (end_prediction_time - start_prediction_time)

    model_accuracies[len(neural_networks_for_comparison)+1] /= iterations
    training_time[len(neural_networks_for_comparison)+1] /= iterations
    prediction_time[len(neural_networks_for_comparison)+1] /= iterations

    print('average accuracy: ', model_accuracies[len(neural_networks_for_comparison)+1])
    print('average training time: ',training_time[len(neural_networks_for_comparison)+1])
    print('average prediction time: ', prediction_time[len(neural_networks_for_comparison)+1])

    np.savetxt('compare_{}_ml.txt'.format(len(neural_networks_for_comparison)+1), [model_accuracies[len(neural_networks_for_comparison)+1],training_time[len(neural_networks_for_comparison)+1],prediction_time[len(neural_networks_for_comparison)+1]], delimiter='\n', fmt='%f')

    np.savetxt("algorithms_comparison_accuracies.txt", model_accuracies, delimiter='\n', fmt='%f')
    np.savetxt("algorithms_comparison_training_time.txt", training_time, delimiter='\n', fmt='%f')
    np.savetxt("algorithms_comparison_predicting_time.txt", prediction_time, delimiter='\n', fmt='%f')
    
if __name__ == '__main__':
    num_of_args = len(sys.argv)

    if num_of_args == 1:
        compare_algorithms_accuracy_time(1)
    elif num_of_args == 3:
        compare_algorithms_accuracy_time(1,True,sys.argv[1], sys.argv[2])
    else:
        print('Wrong amount of arguments: \n for user data tests enter: py -m testing.algorithm_comparison_mnist <folder with pictures> <file with class labels> \n for MNIST data tests enter: py -m testing.algorithm_comparison_mnist')