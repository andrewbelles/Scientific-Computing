import matplotlib.pyplot as plt 
import numpy as np 
import argparse 

def process_1d_data(path):

    data = []

    with open(path, "r", encoding='utf-8') as file_object:
        for line in file_object:
            x = float(line.strip())
            data.append(x)
    return np.array(data)

def process_2d_data(path):

    data = []

    with open(path, "r", encoding='utf-8') as file_object:
        for line in file_object:

            x, y = map(float, line.split(','))
            data.append([x, y])
    return np.array(data)

def plot_test_data(x, y, title="Data Plot"): 

    plt.scatter(x, y)
    plt.xlabel("Time [0:20*pi]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.savefig(title + ".png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Epoch File")

    args = parser.parse_args()

    input_path = "test_data.txt"
    epoch_data = "output_files/" + args.input

    # Vec is the input and expected output
    vec    = process_2d_data(input_path)
    output = process_1d_data(epoch_data)

    # Grab inputs 
    inputs = vec[:, 0]

    title = args.input
    plot_test_data(inputs, output, title)

if __name__ == "__main__":
    main()
