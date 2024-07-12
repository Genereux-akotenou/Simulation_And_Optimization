

# Import requirements
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize shared comm utils
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Initial condition
def f(x):
    if (300 <= x <= 400):
        return 10
    return 0

# Bar Length & speed
L = 1000
c = 1

# Space mesh
nx = 200
dx = L / (nx-1)
x = np.linspace(0, L, nx)

# Time mesh
CFL = 1
nt = 128
dt = CFL * (dx/abs(c))

# Init
u = np.array([f(x[i]) for i in range(nx)])
un = np.zeros(nx)
    
# Solver function
def solve_1d_linearconv(u, un, nt, nx, dt, dx, c):
    # Turn on interactive mode
    plt.ion()  

    for n in range(nt):  
        # Iterrative update
        un = u.copy()
        for i in range(1, nx): 
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
            
        # Boundaries conditions
        if RANK < SIZE-1:
            COMM.send(u[-1], dest=RANK+1, tag=2)
        if RANK != 0:
            u[0] = COMM.recv(source=RANK-1, tag=2)

        # Simul realtime
        COMM.send(u, dest=0, tag=1)
        if RANK == 0:
            # Collect and join all chunch of u
            u_final = np.array([])
            for process in range(SIZE):
                u_chunck = COMM.recv(source=process, tag=1)
                u_final = np.concatenate((u_final, u_chunck)) if process == 0 else np.concatenate((u_final, u_chunck[1:]))
            
            # Plot u
            plt.clf()
            plt.grid()
            plt.plot(x, u_final, '-')
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title("1D Linear Convection Visualization")
            plt.pause(0.001)

    plt.ioff()
    plt.show()

    return 0

# ------------------------------------------------
# Parallel paradigm: Let solve and plot solution
# ------------------------------------------------

assert SIZE <= nx and SIZE >= 2, "The number of process should not be greater than total mesh siize"

if RANK == 0:
    exec_time = time.time()

    # Let Split u array
    r = nx % SIZE
    u_array = np.split(u[r:], SIZE)
    u_array[0] = np.concatenate((u[:r], u_array[0]))
    for i in range(1, len(u_array)):
        u_array[i] = np.insert(u_array[i], 0, u_array[i-1][-1])
    
    # Send u to each process
    for process in range(SIZE):
        COMM.send(u_array[process], dest=process, tag=0)

# Wait for u and send result to master
u = COMM.recv(source=0, tag=0)
nx = len(u)
un = np.zeros(nx)
solve_1d_linearconv(u, un, nt, nx, dt, dx, c)

if RANK == 0:
    print(F"Execution time: {time.time() - exec_time}s")

