
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

"""
IMPORTANT: We must copy "utils" file provided for poisson exercise in the current folder
"""
from utils import compute_dims

# Initialize comm utils
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class GameOfLife:
    def __init__(self, max_transition, grid_row, grid_col, gui, pause_interval):
        # Genenral setup
        self.max_transition = int(max_transition)
        self.rows  = int(grid_row)
        self.cols  = int(grid_col)
        self.gui   = gui
        self.speed = pause_interval

        # Setup Cartesian topology
        self.dims, self.blocksizes = compute_dims(SIZE, [self.rows, self.cols])
        self.cart2d  = COMM.Create_cart(dims=self.dims, periods=None, reorder=False)

        # Global Grid & local-ghost Grid
        self.local_grid = np.zeros((self.blocksizes[0] + 2, self.blocksizes[1] + 2))
        self.local_grid[1:-1, 1:-1] = np.random.randint(2, size=(self.blocksizes[0], self.blocksizes[1]))
        if RANK == 0:
            self.global_grid  = np.zeros((self.dims[0]*self.blocksizes[0], self.dims[1]*self.blocksizes[1]))
    
    def apply_rules(self):
        # Exchange boundary information
        TOP_PROCESS, BOTTOM_PROCESS = self.cart2d.Shift(direction=0, disp=1)
        LEFT_PROCESS, RIGHT_PROCESS = self.cart2d.Shift(direction=1, disp=1)

        self.cart2d.Sendrecv(self.local_grid[1, 1:-1], dest=TOP_PROCESS, recvbuf=self.local_grid[-1, 1:-1], source=BOTTOM_PROCESS)
        self.cart2d.Sendrecv(self.local_grid[-2, 1:-1], dest=BOTTOM_PROCESS, recvbuf=self.local_grid[0, 1:-1], source=TOP_PROCESS)
        self.cart2d.Barrier()

        recv_right_col = np.zeros(self.blocksizes[0]+2)
        self.cart2d.Sendrecv(np.ascontiguousarray(self.local_grid[:, 1]), dest=LEFT_PROCESS, recvbuf=recv_right_col, source=RIGHT_PROCESS)
        self.local_grid[:, -1] = recv_right_col

        recv_left_col = np.zeros(self.blocksizes[0]+2)
        self.cart2d.Sendrecv(np.ascontiguousarray(self.local_grid[:, -2]), dest=RIGHT_PROCESS, recvbuf=recv_left_col, source=LEFT_PROCESS)
        self.local_grid[:, 0] = recv_left_col

        # Extract neighbors submatrix & apply game rules
        local_grid_copy = self.local_grid.copy()
        for i in range(1, self.blocksizes[0]+1):
            for j in range(1, self.blocksizes[1]+1):
                neighbors_matrix = local_grid_copy[i-1:i+2, j-1:j+2]
                neighbors_alive  = np.sum(neighbors_matrix) - local_grid_copy[i, j]
                if local_grid_copy[i, j] == 1:
                    if neighbors_alive < 2 or neighbors_alive > 3:
                        self.local_grid[i, j] = 0 
                    else:
                        self.local_grid[i, j] = 1
                else: 
                    if neighbors_alive == 3:
                        self.local_grid[i, j] = 1
    
    def visu(self, i):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            cmap = plt.matplotlib.colors.ListedColormap(['gray', 'yellow'])
            for i in range(self.rows+1):
                self.ax.axhline(i-0.5, color='#939292', lw=0.5)
            for j in range(self.cols+1):
                self.ax.axvline(j-0.5, color='#939292', lw=0.5)
            self.img = self.ax.imshow(self.global_grid, cmap=cmap)
        else:
            self.img.set_data(self.global_grid)

        self.ax.set_title(f"Iteration: {i+1}/{self.max_transition} - Over: {SIZE} process - Board: ({self.rows}x{self.cols})")
        plt.draw()
        if self.speed != 0:
            plt.pause(self.speed)

    def play(self):
        for i in range(self.max_transition):
            # next state
            self.apply_rules()

            # Gather new state
            COMM.Barrier()
            all_local_grid = COMM.gather(self.local_grid , root=0)
            if RANK == 0:
                for j in range(SIZE):
                    coord = self.cart2d.Get_coords(j)
                    row_start = coord[0]*self.blocksizes[0]
                    row_end   = coord[0]*self.blocksizes[0] + self.blocksizes[0]
                    col_start = coord[1]*self.blocksizes[1]
                    col_end   = coord[1]*self.blocksizes[1] + self.blocksizes[1]
                    self.global_grid[row_start:row_end, col_start:col_end] = all_local_grid[j][1:-1,1:-1]
                print(f"transition: ", i+1)
                print(self.global_grid)

                # Visualization
                if self.gui:
                    self.visu(i)
        plt.show()

# Main
game = GameOfLife(max_transition=200, grid_row=55, grid_col=50, gui=True, pause_interval=0.3)
game.play()
