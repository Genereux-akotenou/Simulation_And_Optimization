#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:50:55 2024

@author: genereux
"""

import numpy as np
import matplotlib.pyplot as plt

# Dimenssions du de la plaque
Lx = 1000
Ly = 100

# Nombre de neuds
Nx = 30
Ny = 30

# Pas du maillage
dx, dy = Lx / (Nx-1), Ly / (Ny-1)

# Maillage de la plaque
x = np.zeros((Nx, Ny))
y = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        x[i, j] = i*dx
        y[i, j] = j*dy

# Initialization : pas de polluant 
U = np.zeros((Nx, Ny))

# Initialisation de Un+1
Unew = np.zeros((Nx, Ny))

# -----
temps = 0
tempsArret = 2000
a1 = 10
a2 = 0.5
D = 9

# Condition de stabilité pour le transport et la diffusion
CFL = 0.8
dt1 = CFL*min(dx, dy) / max(abs(a1), abs(a2))
dt2 = CFL*min(dx**2, dy**2) / (2*D)
dt = min(dt1, dt2)

# Coefficient
lamda1 = (dt*a1) / dx
lamda2 = (dt*a2) / dy
beta1 = (dt*D) / dx**2
beta2 = (dt*D) / dy**2

# Boucle perinciale de resolution
while (temps < tempsArret):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            Unew[i, j] = U[i,j] - lamda1*(U[i,j] - U[i-1,j]) \
                                - lamda2*(U[i,j] - U[i,j-1]) \
                                + beta1*(U[i-1,j] - 2*U[i,j] + U[i+1,j]) \
                                + beta2*(U[i,j-1] - 2*U[i,j] + U[i,j+1])
            
    # Sud
    Unew[:,0]    = Unew[:,1]
    # Nord
    Unew[:,Ny-1] = Unew[:,Ny-2]
    # Est
    Unew[Nx-1,:] = Unew[Nx-2,:]
    # Ouest
    for j in range(Ny):
        if (40 <= y[0,j] <= 60):
            Unew[0,j] = 1
        else:
            Unew[0,j] = 0
            
    # Updates
    temps += dt
    U = Unew.copy()

    # Plot
    plt.contourf(x, y, U, 20)
    plt.colorbar()
    plt.show()

