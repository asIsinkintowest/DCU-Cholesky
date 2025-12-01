import numpy as np

def genMatrix():
    A = np.random.randn(32,32).astype(np.float32)
    np.savetxt("data/data1.csv", A, delimiter=",")

    B = np.random.randn(32,32).astype(np.float32)
    np.savetxt("data/data2.csv", A, delimiter=",")

if __name__ == "__main__":
    genMatrix()
