
from mpi4py import MPI
import numpy as np
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

assert SIZE >= 1, "The number of process should be at least 1"

# Set Problem Data
n = 300

# Initialize matrix on process 0 and scatter part
if RANK == 0:
    matrix = np.random.rand(n, n)
    diag_element = np.diag(matrix)
    diag_array   = np.array_split(diag_element, SIZE)
else:
    diag_array = None

sub_diag  = COMM.scatter(diag_array, root=0)
sub_trace = np.sum(sub_diag)

# Process 0 get all sub trace and sum it
COMM.Barrier()
glob_trace = COMM.reduce(sub_trace, op=MPI.SUM, root=0)
if RANK == 0:
    print(f"Parralel Trace = {glob_trace}")
    print(f"Numpy Trace = {np.trace(matrix)}")
