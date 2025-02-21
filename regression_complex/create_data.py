import argparse 
import random
import sys 
import numpy as np


def create_test_data(size):
    x = np.linspace(0, 10*np.pi, num=size)
    y = np.sin(x)

    return np.stack((x, y)) 

def write_data(filename, data):

    with open(filename, "w") as f:
        for i in range(data.shape[1]):
            f.write(f"{data[0, i]},{data[1, i]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("size", type=int)

    args = parser.parse_args()

    data = create_test_data(args.size)
    write_data(args.filename, data)

if __name__ == "__main__":
    main()
