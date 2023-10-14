import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

from knn_algorithm import KNN

def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], -1)
    x = x.astype("float32") / 255
    return (x[:limit]), (y[:limit])

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

model = KNN()
model.fit(x_train, y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
print(acc)