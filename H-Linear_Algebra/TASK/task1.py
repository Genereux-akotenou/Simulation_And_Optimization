# Utils
import numpy as np
from scipy.io import mmread
import time
import matplotlib.pyplot as plt

def plot_numeric_complexity(path="matrix_set/bcspwr06.mtx"):
    plt.ion()

    # Load the matrix from the file
    matrix = mmread('matrix_set/bcspwr06.mtx')
    matrix = matrix.toarray()

    m, n = matrix.shape
    if m != n:
        raise Exception("Non squared matrix")
    
    solving_time = []
    bloc_size = []
    bloc = 1
    for bloc in range(1, m//2, 2 * bloc):
        A = matrix[:bloc, :bloc]
        print(f'bloc size: {bloc}x{bloc}')
        start = time.time()
        Q, R = np.linalg.qr(A)
        #x = np.linalg.solve(A, b)
        end = time.time()
        bloc_size.append(bloc)
        solving_time.append(end-start)

        # Plot
        plt.clf()
        plt.plot(bloc_size, solving_time, '-')
        #plt.grid()
        plt.xlabel('Matrix bloc size')
        plt.ylabel('time')
        plt.title("QR DECOMPOSITION BY BLOCK")
        plt.pause(0.001)

    plt.ioff()
    plt.show()

plot_numeric_complexity()
