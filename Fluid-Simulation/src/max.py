
import numpy as np 

def read():

    data = np.loadtxt('densities.txt', delimiter=',')

    return data[:, 0], data[:, 1]


def main():
    
    hi, lo = read()
    max_hi = np.max(hi)
    max_lo = np.max(lo)

    print(f"Range [{max_lo} {max_hi}]")


if __name__ == "__main__":
    main()
