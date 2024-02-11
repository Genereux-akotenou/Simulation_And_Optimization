#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:05:45 2024

@author: genereux
"""


import numpy as np
import matplotlib.pyplot as plt

# Condition initiale
def f(x):
    if (300 <= x <= 400):
        return 10
    return 0

# Dimenssions(Longeur) de la barre
L = 1000

# Nombre de neuds
N = 200

# Pas du maillage
dx = L / (N-1)

# Maillage de la barre
x = np.linspace(0, L, N)

# Conditions limites
U = np.array([f(x[i]) for i in range(N)])


# Temps inital
temps = 0
tempsArret = 2000

# Coefficient de diffusion & Vitesse de diffusion & Stabilité
D = 10 
a = 1 
CFL = 0.8

# Coef
dt1 = CFL * (dx/abs(a))
dt2 = CFL * dx**2 /(2*D)
dt = min(dt1, dt2)
dt = CFL * min((2*D)/a**2, dx**2/(2*D))
beta = (dt * D) /  (dx**2)
lamba = dt * (a/dx)

Unew = np.zeros(N)

while (temps < tempsArret):
    for i in range(1, N-1):
        Unew[i] = U[i] - lamba*(U[i] - U[i-1])  + beta*(U[i-1] - 2*U[i] + U[i+1])
        
    Unew[0] = Unew[N-2]
    Unew[N-1] = Unew[N-2]
    temps += dt
    U = Unew.copy()
    plt.grid()
    plt.plot(x, U, '-r')
    plt.axis([0, 1000, 0, 10])
    plt.pause(0.01)


