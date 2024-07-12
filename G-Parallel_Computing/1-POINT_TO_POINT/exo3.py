
from mpi4py import MPI

COMM = MPI.COMM_WORLD 
SIZE = COMM.Get_size() 
RANK = COMM.Get_rank()

if SIZE != 2 and RANK == 0:
    print("This program requires exactly 2 processes.")
    COMM.Abort()

num_process = (RANK+1)%2
shared_variable = 0

for iter in range(10):
    #Ping-Pong
    if RANK == iter % 2:
        COMM.send(shared_variable, dest=num_process)
    else:
        received_data = COMM.recv(source=num_process)
        shared_variable = received_data + 1
        print(f"Process {RANK} got {received_data} from Process {num_process}")
