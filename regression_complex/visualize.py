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

    print(title)

    plt.scatter(x, y)
    plt.xlabel("Time [0:20*pi]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.savefig(title + ".png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path")
    parser.add_argument("expected")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    args = parser.parse_args()

    input_path = args.expected 
    base_epoch_data = "output_files/" + args.base_path
    png_path = "images/" + args.base_path

    # Vec is the input and expected output
    vec    = process_2d_data(input_path)

    # Grab inputs 
    inputs   = vec[:, 0]
    expected = vec[:, 1]

    for epoch in range(args.start, args.end+1):
        epoch_path = f"{base_epoch_data}{epoch:04d}.txt"
        
        title = f"{png_path}{epoch:04d}"
        output = process_1d_data(epoch_path)
        plot_test_data(inputs, output, title=title)

    plot_test_data(inputs, expected, "images/expected_output")

if __name__ == "__main__":
    main()
