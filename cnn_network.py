import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from dense_layer import Dense
from convolutional_layer import Convolutional
from reshape_layer import Reshape
from activation_functions import Leaky_Relu, Sigmoid, Softmax
from loss_functions import binary_cross_entropy, binary_cross_entropy_prime, categorical_cross_entropy, categorical_cross_entropy_prime
from network import train, predict

def preprocess_data(x, y, limit):
    # zero_index = np.where(y == 0)[0][:limit]
    # one_index = np.where(y == 1)[0][:limit]
    # all_indices = np.hstack((zero_index, one_index))
    # all_indices = np.random.permutation(all_indices)
    # x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x[:limit], y[:limit]

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

# neural network
network = [
    Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5),
    Leaky_Relu(),
    Reshape(input_shape=(5, 26, 26), output_shape=(5 * 26 * 26, 1)),
    Dense(input_size=5 * 26 * 26, output_size=100),
    Sigmoid(),
    Dense(input_size=100, output_size=10),
    Softmax()
]

# train
train(
    network,
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test (MNIST test data)
correct_predictions = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    if(np.argmax(output) == np.argmax(y)):
        correct_predictions += 1
print("accuracy rate: ", correct_predictions/len(x_test))