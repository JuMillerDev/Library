import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot(subplot_files:[str],subplot_names:[str],result_file_name:str, plot_name:str, y_axis_name: str, x_axis_name: str, show_plot: bool = False):
    plt.figure(figsize=(10, 6))
    max_length_x = 0

    # Iterate through subplot files and names
    for file, name in zip(subplot_files, subplot_names):

        with open(file, 'r') as f:
            data = [float(line.strip()) for line in f.readlines()]

        x_values = range(1, len(data) + 1)
        plt.plot(x_values,data, label=name)

        max_length_x = max(max_length_x, len(data))

    plt.xticks(range(1, max_length_x + 1))

    if len(subplot_files) > 1:
        plt.legend()

    plt.title(plot_name)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)

    plt.savefig(result_file_name)

    if(show_plot):
        plt.show()

def plot_confusion_matrix_mnist(matrix, result_file_name:str, plot_name:str, y_axis_name: str, x_axis_name: str, show_plot: bool = False):
    # Ensure there are no zero sums in the matrix for user input
    row_sums = matrix.sum(axis=1)
    zero_sum_rows = row_sums == 0
    matrix[zero_sum_rows, :] = 1  # Set all elements in zero sum rows to 1 to avoid division by zero

    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    matrix[zero_sum_rows, :] = 0

    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    
    plt.xticks([x + 0.5 for x in range(10)],range(10))
    plt.yticks([y + 0.5 for y in range(10)],range(10),rotation=0)
    
    plt.title(plot_name)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)

    plt.savefig(result_file_name)

    if(show_plot):
        plt.show()

def save_model( model,file_name: str = 'model'):
    with open('{}.pkl'.format(file_name), 'wb') as file:
        pickle.dump(model,file)

def load_model(file_name: str = 'model'):
    with open('{}.pkl'.format(file_name),'rb') as file:
        return pickle.load(file)