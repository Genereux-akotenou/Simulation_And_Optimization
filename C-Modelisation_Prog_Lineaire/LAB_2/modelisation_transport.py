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

a = 1
temps = 0
tempsArret = 2000
CFL = 1

dt = CFL * (dx/abs(a))
lamba = dt * (a/dx)

Unew = np.zeros(N)

while (temps < tempsArret):
    for i in range(1, N-1):
        # Décentré amont
        #Unew[i] = U[i] - lamba*(U[i] - U[i-1]) 
        
        # Décentré aval
        #Unew[i] = U[i] - lamba*(U[i+1] - U[i])
        
        # Schema centré
        #Unew[i] = U[i] - (lamba/2)*(U[i+1] - U[i-1]) 
        
        # Schema de Lax-Friedrich
        #Unew[i] = (U[i-1] + U[i+1])/2 - (lamba/2)*(U[i+1] - U[i-1]) 
        
        # Schema Lax-Wendroff
        beta = 1
        Unew[i] = U[i] - (lamba/2)*(U[i+1] - U[i-1]) + beta*(lamba**2/2)*(U[i-1] - 2*U[i] + U[i+1])
        
    Unew[0] = Unew[N-1] # Periodique
    Unew[N-1] = Unew[N-2]
    temps += dt
    U = Unew.copy()
    #plt.figure()
    plt.grid()
    plt.plot(x, U, '-r')
    plt.pause(0.01)
    
    
# POUR CFL moins de 1 on a de la diffusion: on a pproximé dux  par ui-ui_1/dx. la diffusion vient de l'erreur de troncateur d'ordre 2. une diffusion numerique non physique
# theroeme de lax: stabilité + consistence = convergence
# L'instabilité c'est quand les oscillation tendent vers l'infinis; si on a des oscillation minim par moment on peut dire que le schema est stable
# Quand on augment l'ordre pour plus de precesion et on a de petite oscillation les gens introduisent de la diffusion pour compenser cet effet
# 