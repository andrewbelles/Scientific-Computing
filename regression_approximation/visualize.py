import matplotlib.pyplot as plt 
import numpy as np 
import re

def process_test_data(path):

    data = []

    with open(path, "r", encoding='utf-8') as file_object:
        for line in file_object:

            x, y = map(float, line.split(','))
            data.append([x, y])

    vec = np.array(data)
    return vec

def plot_test_data(vec): 

    x = vec[:, 0]
    y = vec[:, 1]

    plt.scatter(x, y)
    plt.xlabel("Time [0:2pi]")
    plt.ylabel("Amplitude")
    plt.title("Randomized Data")
    plt.savefig("randomized_data.png")

path = "output.txt"
vec = process_test_data(path)
plot_test_data(vec)

