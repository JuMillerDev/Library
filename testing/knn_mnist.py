import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

from machine_learning_algorithms.knn.knn_algorithm import KNN

def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], -1)
    x = x.astype("float32") / 255
    return (x[:limit]), (y[:limit])

def test_network(predicted, y_test):
        # test (MNIST test data)
    correct_predictions = 0
    for pred, y in zip(predicted, y_test):
        if(pred == np.argmax(y)):
            correct_predictions += 1
    return correct_predictions/len(predicted)


def knn_metrics_k_test(iterations: int = 2):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)
    x_test, y_test = preprocess_data(x_test, y_test, 1000)

    range_of_k = 13

    for k in range(1, range_of_k+1):
        model = KNN()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
    # acc = accuracy_score(y_test, pred)
        print('before')
        acc2 = test_network(pred, y_test)
        print('after')
        print(acc2)
    # print(acc)


if __name__ == "__main__":
    knn_metrics_k_test()
