"""
Exercise 1: Parallel Monte Carlo for PI
* Implement a parallel version of Monte Carlo using the function above:
* Ensure your program works correctly if N is not an exact multiple of the number of processes P
"""

# Import requirements
from mpi4py import MPI
import numpy as np
import random
import time

# Initialize shared comm utils
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Function
def compute_points(INTERVAL):
    random.seed(42)  
    circle_points= 0

    # Total Random numbers generated= possible x 
    # values* possible y values 
    for i in range(INTERVAL**2): 
        rand_x= random.uniform(-1, 1) 
        rand_y= random.uniform(-1, 1) 
      
        # Distance between (x, y) from the origin 
        origin_dist=np.sqrt(rand_x**2 + rand_y**2)
      
        # Checking if (x, y) lies inside the circle 
        if origin_dist<= 1: 
            circle_points+= 1
    
    return circle_points

# Data
N = 3250

# MAIN
if RANK == 0:
    t = time.time()
    sliced_interval = [N//SIZE for _ in range(SIZE)]
    sliced_interval[0] = sliced_interval[0]+(N % SIZE)
    sum_of_point = sum([i**2 for i in sliced_interval])
else:
    sliced_interval = None

recvbuff = COMM.scatter(sliced_interval, root=0)
local_circle_points = compute_points(recvbuff)

COMM.Barrier()

circle_points = COMM.reduce(local_circle_points, op=MPI.SUM, root=0)
if RANK == 0:
    print(f"Total circle points: {circle_points} OVER {sum_of_point} points")
    print(f"[TIME]: {time.time()-t}s")
    print(f"[PI  ]: {4*circle_points/sum_of_point}")