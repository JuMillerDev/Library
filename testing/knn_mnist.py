import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import cv2
import sys
import os
import matplotlib.pyplot as plt

from machine_learning_algorithms.knn.knn_algorithm import KNN
from testing.utils import plot

def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], -1)
    x = x.astype("float32") / 255
    return (x[:limit]), (y[:limit])

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

    return (images_array,labels)


def knn_metrics_k_test(iterations: int = 2, user_test_data: tuple = None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 10000)
    
    if user_test_data == None:
        x_test, y_test = preprocess_data(x_test, y_test, 1000)
    else:
        x_test, y_test = user_test_data[0], user_test_data[1]

    range_of_k = 13

    tested_metrics = [  
                        ('Euklidesowa','euc'),
                        ('Manhattan','man'),
                        ('Chebyszewa', 'cheb'),
                        ('Kosinusowa', 'cosine')
                    ]

    for index,metric in enumerate(tested_metrics):
        print("distance metric: {}/{}".format(index+1,len(tested_metrics)))
        accuracies = np.zeros(range_of_k)
        for k in range(1, range_of_k+1):
            print("K: ", k)
            for i in range(iterations):
                print("Iteration: {}/{}".format(i+1,iterations))
                model = KNN(K=k, distance_measure=metric[1])
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                acc = accuracy_score(y_test, pred)
                print('testing accuracy: ',acc)
                accuracies[k-1] += acc
            accuracies[k-1] /= iterations
        file_name = "knn_distances_{}".format(metric[1])
        np.savetxt("{}.txt".format(file_name), accuracies, delimiter='\n', fmt='%f')
        plot(["{}.txt".format(file_name)],[''],'knn_plot_{}'.format(metric[1]),"Poprawność modelu dla różnych K: metryka {}".format(metric[0]),'Poprawność','Parametr K')
        

if __name__ == "__main__":
    num_of_args = len(sys.argv)
    
    if num_of_args == 1:
        knn_metrics_k_test(1)
    elif num_of_args == 3:
        user_data = preprocess_user_data(sys.argv[1], sys.argv[2])
        knn_metrics_k_test(1,user_data)
    else:
        print('Wrong amount of arguments: \n for user data tests enter: py -m testing.knn_mnist <folder with pictures> <file with class labels> \n for MNIST data tests enter: py -m testing.knn_mnist')

