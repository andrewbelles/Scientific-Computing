import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import griddata
import numpy as np 

import argparse 
import glob
import os

# Takes in a path to data and the number of expected inputs/outputs
def process_data(path, expected_io_count=None):
    data = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            elements = list(map(float, line.strip().split(',')))
            if expected_io_count != len(elements):
                raise ValueError(f"Expected {expected_io_count} values, got {len(elements)}")
            data.append(elements)

    return np.array(data)


def plot_data(inputs, outputs, title="Data Plot", input_dim=0, output_dim=0): 

    fig = plt.figure(figsize=(10,6))

    # Reshape to column if 1d 
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)


    if input_dim + output_dim == 2:

        x_label = f'Input Dimension {input_dim}' if inputs.shape[1] > 1 else 'Input'
        y_label = f'Output Dimension {output_dim}' if outputs.shape[1] > 1 else 'Output'

        plt.scatter(inputs[:, input_dim-1], outputs[:, output_dim-1], alpha=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    
    elif input_dim + output_dim == 3:


        if inputs.ndim == 1:
            x = inputs[:, 0]
            y = outputs[:, 1]
            z = outputs[:, 2]
        elif inputs.ndim == 2:
            x = inputs[:, 0]
            y = inputs[:, 1]
            z = outputs[:, 0]


        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)

        try:
            Z = griddata((x, y), z, (X, Y), method='linear')

            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
        except Exception as e:
            print(f"Error {e}")
            plt.close(fig)
            return

    plt.title(title)
    plt.grid(True)

    tag = f"images/{title}.png"
    plt.savefig(tag, bbox_inches='tight')
    plt.close(fig)
    print(tag)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("expected")
    parser.add_argument("--dims", nargs=2, type=int, default=[1,1])
    parser.add_argument("--range", nargs=2, type=int, default=[-1,-1])
    args = parser.parse_args()

    input_dims  = args.dims[0]
    output_dims = args.dims[1]
    expected_io = input_dims + output_dims

    # Clear image folder 
    pattern = os.path.join("images/", "*.png")
    to_delete = glob.glob(pattern)

    for file in to_delete:
        os.remove(file)
        print(f"Deleted {file}")

    # Load data expected from command line inputs 
    loaded_data = process_data(args.expected, expected_io)
    inputs      = loaded_data[:, :input_dims]
    expected    = loaded_data[:, input_dims:]
    print("Processed Data")

    plot_data(inputs, expected, "expected_output", input_dims, output_dims)

    # Only plot epoch data if expected 
    if args.range[0] != -1:
        for epoch in range(args.range[0], args.range[1]+1):
            epoch_path = f"outputs/epoch_{epoch:04d}.txt"
            
            network_outputs = process_data(epoch_path, output_dims)

            plot_data(inputs, network_outputs, f"epoch_{epoch:04d}", input_dims, output_dims)

if __name__ == "__main__":
    main()
