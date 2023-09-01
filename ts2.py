#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:10:25 2023

@author: feer
"""
# #%%
import numpy as np
import matplotlib.pyplot as plt


def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    end = nn / fs
    tt  = np.linspace(start=0, stop=end, num = nn, endpoint = False)

    xx = vmax * np.sin(2. * np.pi * ff * tt + ph) + dc
    
    return tt,xx

def mi_funcion_DFT( xx_t ):
    N = len(xx_t)
    XX_f = np.zeros(N)+0j
    for k in range(N):
        omega = 2*np.pi * k / N
        XX_f[k] = sum(xx_t[n] * np.exp(-1j * omega * n) for n in range(N))
    return XX_f

#    for k in range(N):
#        for n, x in enumerate(xx_t):
#            XX_f[k] += x * np.exp(-1j*2*np.pi * k*n/N)

#    for k in range(0,N):
#        for n in range(0,N):
#            X_f[k] += xx[n] * np.exp(-1j*2*np.pi*k*n/N)


def graficar_espectro_fft(signal):
	df = fs/N
	ff = np.linspace(0, (N-1)*df, N)
	ft_XX = 1/N * np.fft.fft( signal, axis = 0 )
	bfrec = ff <= fs/2
	plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_XX[bfrec])**2), \
	          ':', label='FFT', linewidth=2)

def graficar_espectro_dft(signal):
	df = fs/N
	ff = np.linspace(0, (N-1)*df, N)
	ft_XX = 1/N * mi_funcion_DFT( signal )
	bfrec = ff <= fs/2
	plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_XX[bfrec])**2), \
	          '-', label='mi DFT', linewidth=1)


# Defino los parámetros de la simulación
fs = 1000 # Hz
N  = fs  # n° de muestras

delta_f = fs / N # Hz

Ts = 1 / fs # segundos
T_simulacion = N * Ts # segundos


# Defino los parámetros de mi señal
fx = 10 # Hz
Ax = np.sqrt(2) # Volts - la amplitud sqrt(2) me dará una potencia unitaria
offset = 0 # Volts
phase = 0


# Genero mi señal senoidal con los parámetros anteriores
t, x_t  = mi_funcion_sen(Ax, offset, fx, phase, N, fs)

snr_dB = -10.*np.log10(2**3) # signal to noise ratio, valor deseado
var_noise  = 10 ** (snr_dB / (-10)) # varianza (potencia)

#ruido = np.random.normal(loc=0.0, scale=np.sqrt(var_noise), size=N)
q = np.sqrt(12 * var_noise)
ruido = np.random.uniform(-q/2, q/2, N)
x_t_ruidosa = x_t + ruido


#plt.close('all')
plt.figure(1) # usar figure() sin arg para q se abra siempre uno nuevo
plt.clf()
plt.subplot(2, 1, 1)
plt.title('sen(x) + ruido (tiempo)')
plt.xlabel('t [s]'), plt.ylabel('Amplitud [V]')
plt.plot(t, x_t_ruidosa)
plt.subplot(2, 1, 2)
plt.title('Espectro')
plt.xlabel('f [Hz]'), plt.ylabel('Amplitud [dB]')
graficar_espectro_fft(x_t_ruidosa)
graficar_espectro_dft(x_t_ruidosa)
plt.legend()
plt.show()
