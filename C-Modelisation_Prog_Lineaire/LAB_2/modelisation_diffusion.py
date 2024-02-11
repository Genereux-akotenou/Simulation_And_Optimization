#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:39:03 2023

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

# Coefficient de diffusion
D = 10

# Temps inital
temps = 0
tempsArret = 2000

# Stabilité
CFL = 0.8
dt = CFL * dx**2 /(2*D)
beta = (dt * D) /  (dx**2)

Unew = np.zeros(N)

while (temps < tempsArret):
    for i in range(1, N-1):
        Unew[i] = U[i]  + beta*(U[i-1] - 2*U[i] + U[i+1])
        
    Unew[0] = Unew[1]
    Unew[N-1] = Unew[N-2]
    temps += dt
    U = Unew.copy()
    plt.grid()
    plt.plot(x, U, '-r')
    plt.axis([0, 1000, 0, 10])
    plt.pause(0.01)
    
    
# POUR CFL moins de 1 on a de la diffusion: on a pproximé dux  par ui-ui_1/dx. la diffusion vient de l'erreur de troncateur d'ordre 2. une diffusion numerique non physique
# theroeme de lax: stabilité + consistence = convergence
# L'instabilité c'est quand les oscillation tendent vers l'infinis; si on a des oscillation minim par moment on peut dire que le schema est stable
# Quand on augment l'ordre pour plus de precesion et on a de petite oscillation les gens introduisent de la diffusion pour compenser cet effet
# 