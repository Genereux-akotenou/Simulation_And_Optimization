
# Import requirements
#from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Initialize shared comm utils
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()

# Initial condition
def f(x):
    if (300 <= x <= 400):
        return 10
    return 0

# Bar Length & speed
L = 1000
c = 1

# Space mesh
Nx = 200
dx = L / (Nx-1)
x = np.linspace(0, L, Nx)

# Time mesh
CFL = 1
Nt = 128
dt = CFL * (dx/abs(c))

# Init
u = np.array([f(x[i]) for i in range(Nx)])
un = np.zeros(Nx)
    
# Solver function
def solve_1d_linearconv(u, un, nt, nx, dt, dx, c):
    for n in range(nt):  
        # Iterrative update
        un = u.copy()
        for i in range(1, nx): 
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
            
        # Boundaries conditions
        u[0] = u[nx-1]
        #u[nx-1] = u[nx-2]
    return 0

# ------------------------------------------------
# Sequantial paradigm: Let solve and plot solution
# ------------------------------------------------
solve_1d_linearconv(u, un, Nt, Nx, dt, dx, c)
plt.figure()
plt.grid()
plt.plot(x, u, '-')
plt.show()
