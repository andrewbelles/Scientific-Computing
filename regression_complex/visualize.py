import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import griddata
import numpy as np 
import argparse 
import glob
import os
import threading

def plot_data(inputs, outputs, title="Data Plot", input_dim=0, output_dim=0, prediction=None): 
    # Create a new figure and canvas without using pyplot
    fig = Figure(figsize=(10, 6))
    FigureCanvas(fig)
    
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    
    total_dims = input_dim + output_dim
    if total_dims == 2:

        ax = fig.add_subplot(111)
        
        x_label = f'Input Dimension {input_dim}' if inputs.shape[1] > 1 else 'Input'
        y_label = f'Output Dimension {output_dim}' if outputs.shape[1] > 1 else 'Output'

        ax.scatter(inputs[:, input_dim-1], outputs[:, output_dim-1], alpha=0.5, label="Expected")

        if prediction is not None:
            in_pred  = prediction[:, :input_dim]
            out_pred = prediction[:, input_dim:] 
            ax.scatter(in_pred[:, input_dim-1], out_pred[:, output_dim-1], alpha=0.8, label="Predicted")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend()

    elif total_dims == 3:
        if input_dim == 2 and output_dim == 1:
            x = inputs[:, 0]
            y = inputs[:, 1]
            z = outputs[:, 0]
        elif input_dim == 1 and output_dim == 2:
            x = inputs[:, 0]
            y = outputs[:, 0]
            z = outputs[:, 1]
        else:
            raise ValueError("Invalid input/output dimensions for 3D plot")
        
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        
        try:
            Z = griddata((x, y), z, (X, Y), method='linear')
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')

            if prediction is not None:
                pred_x = prediction[:, 0]
                pred_y = prediction[:, 1]
                pred_z = prediction[:, 2]
                
                Z_pred = griddata((pred_x, pred_y), pred_z, (X, Y), method='linear')
                ax.plot_surface(X, Y, Z_pred, cmap='plasma', alpha=0.5, label='Predicted')
                ax.legend()

        except Exception as e:
            print(f"Error {e}")
            return
    
    fig.suptitle(title)
    tag = f"images/{title}.png"
    fig.savefig(tag, bbox_inches='tight')
    print(tag)

def plot_data_threaded(inputs, outputs, title, input_dim, output_dim):
    plot_data(inputs, outputs, title, input_dim, output_dim)

# Rest of the main() and other functions remain the same...
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("expected")
    parser.add_argument("--dims", nargs=2, type=int, default=[1,1])
    parser.add_argument("--range", nargs=2, type=int, default=[-1,-1])
    parser.add_argument("--preserve", type=bool, default=False)
    parser.add_argument("--prediction")
    args = parser.parse_args()

    input_dims  = args.dims[0]
    output_dims = args.dims[1]
    expected_io = input_dims + output_dims

    # If we want to delete files we don't specify preserve

    # Load data expected from command line inputs 
    loaded_data = process_data(args.expected, expected_io)
    inputs      = loaded_data[:, :input_dims]
    expected    = loaded_data[:, input_dims:]
    print("Loaded Input Data")
    
    prediction = None
    if args.prediction:
        prediction = process_data(args.prediction, input_dims + output_dims)
        print(prediction)

    if args.preserve == False and prediction is None:
        # Clear image folder 
        pattern = os.path.join("images/", "*.png")
        to_delete = glob.glob(pattern)

        for file in to_delete:
            os.remove(file)
            print(f"Deleted {file}")

    plot_data(inputs, expected, "expected_output", input_dims, output_dims, prediction)

    # Only plot epoch data if expected 
    if args.range[0] != -1:
        threads = []  # Store the threads to join later
        for epoch in range(args.range[0], args.range[1]+1):
            epoch_path = f"outputs/epoch_{epoch:04d}.txt"

            network_outputs = process_data(epoch_path, output_dims)

            title = f"epoch_{epoch:04d}"
            thread = threading.Thread(target=plot_data_threaded, args=(inputs, network_outputs, title, input_dims, output_dims))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    main()
