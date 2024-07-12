''' This program compute parallel csc matrix vector multiplication using mpi '''

# Import requirements
import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numpy import array, concatenate
from mpi4py import MPI

# Initialize shared comm utils
COMM     = MPI.COMM_WORLD
nbOfproc = COMM.Get_size()
RANK     = COMM.Get_rank()
seed(42)

def matrixVectorMult(A, b, x):

    row, col = A.shape
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]
    return 0

########################initialize matrix A and vector b ######################
SIZE = 1000
q, r = divmod(SIZE, nbOfproc)
Local_size = np.full(nbOfproc, q, dtype=int)
Local_size[:r] += 1
counts = [SIZE*row_count for row_count in Local_size]
displacements = [sum(counts[:i]) for i in range(nbOfproc)]

if RANK == 0:   
    A = lil_matrix((SIZE, SIZE))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(SIZE))
    A = A.toarray()
    A = array(A)
    b = rand(SIZE)
else :
    A = array([])
    b = None

#####Send b to all procs and scatter A (each proc has its own local matrix#####
b = COMM.bcast(b, root=0)
LocalMatrix = np.zeros(counts[RANK])
COMM.Scatterv([A.flatten(), counts, displacements, MPI.DOUBLE], recvbuf=LocalMatrix, root=0)
LocalMatrix = np.reshape(LocalMatrix, (-1, b.shape[0]))

#####################Compute A*b locally#######################################
LocalX = np.zeros(LocalMatrix.shape[0])
start = MPI.Wtime()
matrixVectorMult(LocalMatrix, b, LocalX)
stop = MPI.Wtime()
if RANK == 0:
    print("CPU time of parallel multiplication is ", (stop - start)*1000)

##################Gather te results ###########################################
X = COMM.gather(LocalX, root=0)

##################Print the results ############################################
if RANK == 0 :
    X = concatenate(X)
    X_ = A.dot(b)
    #print("The result of A*b using dot is :", X_)
    print("The result of A*b using dot is :", np.max(X_ - X))
    print("The result of A*b using parallel version is :", X)