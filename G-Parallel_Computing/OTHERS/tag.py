from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 1:
    status = MPI.Status()
    
    # Wait for a message with tag 0 or tag 1
    for i in range(2):
        data_tag = comm.recv(tag=i)
        print(f"Payload: {data_tag}")
    print("done")

    # data_tag0 = comm.recv(source=0, tag=0)
    # print(f"Process {rank} received message with tag 0: {data_tag0}")
    
    # data_tag1 = comm.recv(source=0, tag=1)
    # print(f"Process {rank} received message with tag 1: {data_tag1}")
    
elif rank == 0:
    # Send messages with different tags to process 1
    comm.send("xxx", dest=1, tag=1)

    # Pause for 5 seconds
    time.sleep(5)

    comm.send("yyy", dest=1, tag=0)

