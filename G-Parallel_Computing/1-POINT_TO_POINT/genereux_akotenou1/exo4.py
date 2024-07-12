
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

assert SIZE > 1, "The number of process should be great than 1"

if RANK == 0:
    # Get input
    data = int(input(""))

    # Send
    COMM.send(data + RANK, dest=1)
    print(f"Process {RANK}: Received data {data}")
else:
    # Receive data
    received_data = COMM.recv(source=RANK - 1)
    print(f"Process {RANK}: Received data {received_data}")

    # Send to next
    if RANK != SIZE - 1:
        COMM.send(received_data + RANK, dest=RANK + 1)
