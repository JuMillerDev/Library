import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import cv2
from keras.datasets import mnist
from sklearn.metrics import accuracy_score,confusion_matrix
from machine_learning_algorithms.decision_forest.forest import Forest
from testing.utils import plot_confusion_matrix_mnist

def preprocess_data(x,y,limit):
    x = x.reshape(x.shape[0], -1)
    x = x.astype("float32") / 255
    return x[:limit], np.atleast_2d(y[:limit]).T

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

        img_reshaped = img_array.reshape((1, 784))

        processed_images.append(img_reshaped)

    # Stack the processed images vertically
    images_array = np.vstack(processed_images)

    with open(labels_file_location, 'r') as labels_file:
        labels = [int(line.strip()) for line in labels_file]

    return (images_array,np.atleast_2d(labels).T)

def df_tree_depth_test(iterations: int = 2, user_test_data: tuple = None):
    tree_depth_range = 10
    accuracies = np.zeros(tree_depth_range)
    list_of_matrices = [np.zeros((10, 10)) for _ in range(10)] 
        
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)

    if user_test_data == None:
        x_test, y_test = preprocess_data(x_test, y_test, 1000)
    else:
        x_test, y_test = user_test_data[0], user_test_data[1]


    for depth in range(1, tree_depth_range+1):
        print("depth: ", depth)
        for i in range(iterations):
            print("Iteration: {}/{}".format(i+1,iterations))

            f = Forest(data=x_train,
            labels=y_train,
            # n_trees=100,
            n_trees=1,
            max_features=40,
            bootstrap_features=True,
            max_depth=depth,
            min_leaf_points=2)

            y_pred = f.predict(x_test)
            acc = accuracy_score(y_test,y_pred)
            conf_matrix = confusion_matrix(y_test,y_pred)

            # for user data
            # Pad the confusion matrix with zeros if the number of classes is less than 10
            if conf_matrix.shape[0] < 10 or conf_matrix.shape[1] < 10:
                padded_matrix = np.zeros((10, 10))
                padded_matrix[:conf_matrix.shape[0], :conf_matrix.shape[1]] = conf_matrix
                conf_matrix = padded_matrix

            accuracies[depth-1] += acc
            list_of_matrices[depth-1] += conf_matrix

        accuracies[depth-1] /= iterations
        list_of_matrices[depth-1] /= iterations
        print('average accuracy: ',accuracies[depth-1])
        plot_confusion_matrix_mnist(list_of_matrices[depth-1],'confusion_matrix_depth{}'.format(depth),'Macierz błędów dla glębokości: {}'.format(depth),'prawdziwe klasy','przewidywane klasy')
    np.savetxt("df_depth_accuracies.txt", accuracies, delimiter='\n', fmt='%f')
        


def df_tree_max_features_test(iterations: int = 2, user_test_data: tuple = None):
    max_features_values = [5, 10, 20, 30, 40, 50, 60] 
        
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)

    if user_test_data == None:
        x_test, y_test = preprocess_data(x_test, y_test, 1000)
    else:
        x_test, y_test = user_test_data[0], user_test_data[1]

    accuracies = np.zeros(len(max_features_values))
    training_time = np.zeros(len(max_features_values))
    list_of_matrices = [np.zeros((10, 10)) for _ in range(10)]

    for index,max_features in enumerate(max_features_values):
        print("number of features: ", max_features)

        for i in range(iterations):
            print("Iteration: {}/{}".format(i+1,iterations))

            start_time = time.time()

            f = Forest(data=x_train,
            labels=y_train,
            n_trees=100,
            max_features=max_features,
            bootstrap_features=True,
            max_depth=6,
            min_leaf_points=2)

            end_time = time.time()

            y_pred = f.predict(x_test)
            acc = accuracy_score(y_test,y_pred)
            conf_matrix = confusion_matrix(y_test,y_pred)

            accuracies[index] += acc
            training_time[index] += (end_time - start_time)
            list_of_matrices[index] += conf_matrix
        
        list_of_matrices[index] /= iterations
        training_time[index] /= iterations
        accuracies[index] /= iterations

        print('average accuracy: ',accuracies[index])
        print('average time: ',training_time[index])


        plot_confusion_matrix_mnist(list_of_matrices[index],'confusion_matrix_features{}'.format(max_features),'Macierz błędów dla maksymalnej ilości cech: {}'.format(max_features),'prawdziwe klasy','przewidywane klasy')
    np.savetxt("df_max_features_accuracies.txt", accuracies, delimiter='\n', fmt='%f')
    np.savetxt("df_max_features_time.txt", training_time, delimiter='\n', fmt='%f')

if __name__ == '__main__':
    num_of_args = len(sys.argv)
    
    if num_of_args == 1:
        df_tree_depth_test(1)
        df_tree_max_features_test(1)
    elif num_of_args == 3:
        user_data = preprocess_user_data(sys.argv[1], sys.argv[2])
        df_tree_depth_test(1, user_data)
        df_tree_max_features_test(1, user_data)
    else:
        print('Wrong amount of arguments: \n for user data tests enter: py -m testing.decision_forest_mnist <folder with pictures> <file with class labels> \n for MNIST data tests enter: py -m testing.decision_forest_mnist')