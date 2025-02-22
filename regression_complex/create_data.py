import argparse 
import random
import sys 
import numpy as np


def create_1D_test_data(size):
    x = []
    for i in range(size):
        new = np.random.uniform(0.0, 10.0)
        x.append(new)

    x = np.array(x)
    y = 2*x-2

    return np.stack((x, y)) 


def create_2D_test_data(size):
    x = []
    y = []
    for i in range(size):
        _x = np.random.uniform(0.0, 10.0)
        _y = np.random.uniform(0.0, 10.0)

        x.append(_x)
        y.append(_y)

    x = np.array(x)
    y = np.array(y)
    z = x**2 - y**2
    return np.stack((x,y,z))


def write_1D_data(filename, data):

    with open(filename, "w") as f:
        for i in range(data.shape[1]):
            f.write(f"{data[0, i]},{data[1, i]}\n")


def write_2D_data(filename, data):

    with open(filename, "w") as f:
        for i in range(data.shape[1]):
            f.write(f"{data[0, i]},{data[1, i]},{data[2, i]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("size", type=int)

    args = parser.parse_args()

    data = create_2D_test_data(args.size)
    write_2D_data(args.filename, data)

if __name__ == "__main__":
    main()
