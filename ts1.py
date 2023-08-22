#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:49:05 2023

@author: Fernando Coda
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    end = nn / fs
    tt  = np.linspace(start=0, stop=end, num = nn, endpoint = False)

    xx = vmax * np.sin(2. * np.pi * ff * tt + ph) + dc
    
    return tt,xx


def mi_funcion_cuadrada(vmax, dc, ff, ph, nn, fs):
    end = nn / fs
    tt  = np.linspace(start=0, stop=end, num = nn, endpoint = False)

#    xx = vmax * np.sign(np.sin(2. * np.pi * ff * tt + ph)) + dc
#    xx = vmax * np.array([1 if math.floor(2*ff*t + ph/np.pi) % 2 == 0 
#                          else -1 
#                          for t in tt]) + dc
    ph = ph / np.pi
    vfloor = np.vectorize(math.floor)
#    xx = 2.0 * (2*vfloor(ff*tt) - vfloor(2*(ff*tt))) + 1
    xx = vmax * ((-1) ** vfloor(2*ff*tt + ph)) + dc
    
    return tt,xx




# Defino los parametros de la simulación
fs = 20 # Hz
N  = fs  # n° de muestras

delta_f = fs / N # Hz

Ts = 1 / fs # segundos
T_simulacion = N * Ts # segundos


# Defino los parametros de mi señal
fx = 2 # Hz
Ax = 2 # Volts
offset = 5 # Volts
phase = np.pi/2.


# Genero dos señales, senoidal y cuadrada, ambas con los mismos parámetros
t1, x1_sen  = mi_funcion_sen(Ax, offset, fx, phase, N, fs)
t2, x2_cuad = mi_funcion_cuadrada(Ax, offset, fx, phase, N, fs)


# Grafico ambas señales superpuestas
plt.close('all')
#plt.figure(1)

plt.plot(t1, x1_sen,  'x--', label='sin(t)')
plt.plot(t2, x2_cuad, 'x--', label='square(t)')

plt.ylabel('Amplitud [V]')
plt.xlabel('Tiempo [s]')
#plt.axis('tight')
plt.grid(True)
plt.legend()

plt.show()



# Ejemplos de grillas temporales
#t1 = np.linspace(0, T_simulacion, num = N)
#t2 = np.linspace(0, T_simulacion, num = N, endpoint = False)
#t  = np.arange(start = 0, stop = T_simulacion, step = Ts)

