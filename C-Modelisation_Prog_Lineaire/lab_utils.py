import numpy as np

def gauss_pivot_v2(A: list = [], b: list = []):
    """
    Solves a linear system of equations using Gauss elimination. We manager case where pivot is null here.

    Args:
    A (numpy.ndarray): Coefficient matrix.
    b (numpy.ndarray): Right-hand side vector.

    Returns:
    x (numpy.ndarray): Solution vector.
    """
    n = len(b)
    x = np.zeros(n)

    for k in range(0, n-1):
        # Let's find for maximum pivot in column k
        row_with_max_pivot = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[row_with_max_pivot, k]):
                row_with_max_pivot = i
        
        A[[k, row_with_max_pivot]] = A[[row_with_max_pivot, k]]
        b[k], b[row_with_max_pivot] = b[row_with_max_pivot], b[k]

        if A[k][k] == 0:
            return np.zeros(n)

        # Let's transform the system into an upper triangular matrix.
        for i in range(k + 1, n):
            pivot = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= pivot * A[k, j]
            b[i] -= pivot * b[k]
            
    # Solving, let compute x
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x