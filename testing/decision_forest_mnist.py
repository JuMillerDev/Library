import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from machine_learning_algorithms.decision_forest1.forest import Forest

def preprocess_data(x,y,limit):
    x = x.reshape(x.shape[0], -1)
    x = x.astype("float32") / 255
    return x[:limit], np.atleast_2d(y[:limit]).T


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

f = Forest(data=x_train,
           labels=y_train,
           n_trees=100,
           max_features=40,
           bootstrap_features=True,
           max_depth=6,
           min_leaf_points=2)

print('debug')


y_pred = f.predict(x_test)
print("accuracy score: ",accuracy_score(y_test,y_pred))