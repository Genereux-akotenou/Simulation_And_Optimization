
from mpi4py import MPI
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import time

"""
IMPORTANT: We must copy "utils" file provided for poisson exercise in the current folder
"""
from utils import compute_dims

# Initialize comm utils
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Utils
def solve_2d_diff(u, un, nt, dt, dx, dy, nu):
    # Assign initial conditions - set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    row, col =u.shape 
    coord     = cart2d.Get_coords(RANK)
    row_start = coord[0]*blocksizes[0]
    row_end   = coord[0]*blocksizes[0] + blocksizes[0]
    col_start = coord[1]*blocksizes[1]
    col_end   = coord[1]*blocksizes[1] + blocksizes[1]
    for i, iter_y in enumerate(range(row_start, row_end)):
        for j, iter_x in enumerate(range(col_start, col_end)):
            if 0.5 <= iter_y*dy <= 1.0 and 0.5 <= iter_x*dx <= 1.0:
                u[1:-1, 1:-1][i, j] = 2

    # Some constante
    c1 = (nu*dt) / dx**2
    c2 = (nu*dt) / dy**2

    # Derived type
    type_line = MPI.DOUBLE.Create_contiguous(u.shape[1]-2)
    type_line.Commit()

    # Fill the update of u
    for n in range(nt):
        un = u.copy()

        # Exchange boundary information
        TOP_PROCESS, BOTTOM_PROCESS = cart2d.Shift(direction=0, disp=1)
        LEFT_PROCESS, RIGHT_PROCESS = cart2d.Shift(direction=1, disp=1)
        cart2d.Sendrecv([un[1, 1:-1], 1, type_line],  dest=TOP_PROCESS, recvbuf=[un[-1, 1:-1], 1, type_line], source=BOTTOM_PROCESS)
        cart2d.Sendrecv([un[-2, 1:-1], 1, type_line], dest=BOTTOM_PROCESS, recvbuf=[un[0, 1:-1], 1, type_line], source=TOP_PROCESS)
        
        right_col_buff = np.zeros(u.shape[0]-2)
        cart2d.Sendrecv(np.ascontiguousarray(un[1:-1, 1]), dest=LEFT_PROCESS, recvbuf=right_col_buff, source=RIGHT_PROCESS)
        un[1:-1, -1] = right_col_buff

        left_col_buff = np.zeros(u.shape[0]-2)
        cart2d.Sendrecv(np.ascontiguousarray(un[1:-1, -2]), dest=RIGHT_PROCESS, recvbuf=left_col_buff, source=LEFT_PROCESS)
        un[1:-1, 0] = left_col_buff
        
        # Update new state
        for i in range(1, nx-1): 
            for j in range(1, ny-1):
                u[i, j] = un[i, j] + c1*(un[i+1, j] - 2*un[i, j] + un[i-1, j]) + c2*(un[i, j+1] - 2*un[i, j] + un[i, j-1])
    
    return 0


###variable declarations
nt = 51
nx = 101
ny = 101 
nu = .05
dx = 2 / (nx -1)
dy = 2 / (ny -1)
sigma = .25
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

# create a 1xn vector of 1's
u = np.ones((ny, nx)) 
un = np.ones((ny, nx))

# ------------------------------------------------
# Parallel paradigm: Let solve and plot solution
# ------------------------------------------------

# a) Let create a cartesian topology
dims, blocksizes = compute_dims(SIZE, [nx, ny])
cart2d  = COMM.Create_cart(dims=dims, periods=None, reorder=False)

# a) We setup new values for nx, ny as domaine change
nx = blocksizes[0]+2
ny = blocksizes[1]+2

# a) We cut u to mach it process gridsize and add ghost_cells
u_local  = np.zeros((blocksizes[0] + 2, blocksizes[1] + 2))
u_local[1:-1, 1:-1] = np.ones((blocksizes[0], blocksizes[1]))
un_local = np.zeros((blocksizes[0] + 2, blocksizes[1] + 2))
un_local[1:-1, 1:-1] = np.ones((blocksizes[0], blocksizes[1]))

# b-c) init u locally and exchage boundary information
solve_2d_diff(u_local, un_local, nt, dt, dx, dy, nu)

# d) Get data on process 0 and display
COMM.Barrier()
all_local_u = COMM.gather(u_local , root=0)
if RANK == 0:
    # Reconstruct initial matrix
    for j in range(SIZE):
        coord = cart2d.Get_coords(j)
        row_start = coord[0]*blocksizes[0]
        row_end   = coord[0]*blocksizes[0] + blocksizes[0]
        col_start = coord[1]*blocksizes[1]
        col_end   = coord[1]*blocksizes[1] + blocksizes[1]
        u[row_start:row_end, col_start:col_end] = all_local_u[j][1:-1,1:-1]
        print(u)

    # Visualization
    fig = pyplot.figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    ax.set_title('3D Surface Plot')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    pyplot.show()
