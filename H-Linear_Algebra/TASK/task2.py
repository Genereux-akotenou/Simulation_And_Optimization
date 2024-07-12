#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:47:55 2024

@author: genereux
"""

import numpy as np

# Define the matrix A
A = np.array([
    [3, 2],
    [2, 3],
    [2, -2]
])
A = np.array([
    [1, 4],
    [3, 2],
    [2, 1              ]
])


# SVD
U, S, VT = np.linalg.svd(A, full_matrices=True)

print("Matrix U:", U)
print("Singular values S:", S)
print("Matrix VT:", VT)

# Check
Iu = np.dot(U, np.transpose(U))
Iv = np.dot(VT, np.transpose(VT))

print("U*UT = ", Iu)
print("VT*V =",  Iv)