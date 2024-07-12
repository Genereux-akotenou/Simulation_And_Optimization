
from mpi4py import MPI
COMM = MPI.COMM_WORLD 
SIZE = COMM.Get_size() 
RANK = COMM.Get_rank()

assert SIZE > 1, "The number of process should be great thant 1"

data = 0
while True:
    # Collect Data and broacast
    if RANK == 0:
        data = int(input(""))
        if data < 0:
            COMM.Abort()
        for process in range(SIZE):
            COMM.send(data, dest=process)
    
    # Print received data
    receiveBuff = COMM.recv(source=0)
    print(f"Process {RANK} got {receiveBuff}")
