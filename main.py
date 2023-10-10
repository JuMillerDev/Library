from cnn import *
from dense_layer import Dense

print("Please choose the neural network\n1) Convolutional Neural Network")
user_input = input("\nYour Answear: ")

match user_input:
    case "1": 
        layer = Dense(2,4)
    case _: print("No Neural Network With Such Name")