#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 21:18:36 2023

@author: feer
"""

import numpy  as np
#import pandas as pd
from pandas import DataFrame
from IPython.display import HTML
from matplotlib import pyplot as plt
from scipy import signal
from astropy.stats import median_absolute_deviation
from astropy.stats import mad_std

def espectro(x_t, fs, N, k):
    df  = fs/N
    f   = np.linspace(0, (N-1)*df, N)
    spec = np.fft.fft( x_t, axis = 0 )
    X_f = 10 * np.log10(np.abs(k/N * spec)**2)
    return X_f, f

def graficar_espectro(X_f, f, fs, *args, **kwargs):   
    bfrec = (f <= fs/2)
    plt.plot(f[bfrec], X_f[bfrec], *args, **kwargs)	

# def graficar_espectro(x_t, fs, N, k, *args, **kwargs):    
#     df  = fs/N
#     f   = np.linspace(0, (N-1)*df, N)
#     bfrec = (f <= fs/2)
#     X_f = 10 * np.log10(np.abs(k/N * np.fft.fft( x_t, axis = 0 ))**2)

#     plt.plot(f[bfrec], X_f[bfrec], *args, **kwargs)	


# median(abs(a - median(a)))
def my_median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))
    
#%%
R = 200  # repeticiones o realizaciones
N = 1000 # nº de muestras

# frec. de sampling normalizada
fs = N
delta_f = fs/N

k = 10 # interpolacion

N_p = k * N
delta_f_p = delta_f / k

# Presentación de ventanas a usar
plt.figure(0).clf()
plt.plot(signal.windows.boxcar(N),  label='boxcar')
plt.plot(signal.windows.flattop(N), label='flattop')
plt.plot(signal.windows.hamming(N), label='hamming')
plt.plot(signal.windows.hann(N),    label='hann')
plt.legend()
plt.title('Ventanas')
plt.xlabel(''), plt.ylabel('Amplitud [V]')
#plt.show()

#%%

SNRs = np.array([ 3, 10 ], dtype=np.float64)
resultados = []

for i, snr_dB in enumerate(SNRs):
    
    fr = np.random.uniform(-1/2, 1/2, size=(1,R)) # cada realizacion tiene una frecuencia

    # Frecuencia central normalizada: pi/2 - es equivalente a fs/4
    W0 = fs/4
    W1 = W0 + fr
    
    var_noise  = 10 ** (snr_dB / (-10)) # varianza (potencia)
    noise = np.random.normal(loc=0.0, scale=np.sqrt(var_noise), size=(N,R))
    
    a1 = np.sqrt(2) # amplitud p/ potencia normalizada
    
    end = N / fs
    
    n = np.linspace(start=0, stop=end, num = N, endpoint = False).reshape((N,1))
    
    # scipi signal window
    # para convertir en matriz usar variable.reshape((N,1))
    # ó variable = np.repeat(variable[0], N, axis=1)
    
    x  = a1 * np.sin(2*np.pi * W1 * n) + noise
    
    # Ventaneo
    xx_boxcar   = x * signal.windows.boxcar(N).reshape((N,1))
    xx_flattop  = x * signal.windows.flattop(N).reshape((N,1))
    xx_hamming  = x * signal.windows.hamming(N).reshape((N,1))
    xx_hann     = x * signal.windows.hann(N).reshape((N,1))
    
    # Interpolación a 10x
    xxx_flattop = np.vstack([xx_flattop, np.zeros([(k-1)*N, R])])
    xxx_boxcar  = np.vstack([xx_boxcar,  np.zeros([(k-1)*N, R])])
    
    #%% Espectros
    
    xx_boxcar_f,  ff  = espectro(xx_boxcar,    fs, N,   1)
    xxx_boxcar_f, fff = espectro(xxx_boxcar,   fs, N_p, k)
    
    xx_flattop_f,  ff  = espectro(xx_flattop,  fs, N,   1)
    xxx_flattop_f, fff = espectro(xxx_flattop, fs, N_p, k)
    
    xx_hamming_f,  ff  = espectro(xx_hamming,  fs, N,   1)
    xx_hann_f,     ff  = espectro(xx_hann,     fs, N,   1)
    
    #%%
    #------------------------------ boxcar -----------------------------------------
    
    plt.figure(10*i+1).clf()
    
    #plt.xticks(np.linspace(0, 500, 5), minor=False)
    plt.xticks(np.linspace(W0-2, W0+2, 5), minor=True)
    plt.grid(which='major', alpha=0.7)
    plt.grid(which='minor', alpha=0.3)
    
    graficar_espectro( xx_boxcar_f, ff,  fs, 'o--g', linewidth=1)#, label='$x_n\'$ (boxcar)')
#    graficar_espectro(xxx_boxcar_f, fff, fs,  'x:r', linewidth=1)#, label='$x_n\'\'$ (boxcar)')
    
    plt.title('$x_n\'$ y $x_n\'\'$ (Boxcar), SNR='+str(int(snr_dB)))
    plt.xlabel('f [Hz]'), plt.ylabel('Amplitud [dB]')
    #plt.legend()
    #plt.show()
    
    #----------------------------- flattop -----------------------------------------
    
    plt.figure(10*i+2).clf()
    
    #plt.xticks(np.linspace(0, 500, 5), minor=False)
    plt.xticks(np.linspace(W0-2, W0+2, 5), minor=True)
    plt.grid(which='major', alpha=0.7)
    plt.grid(which='minor', alpha=0.3)
    
    #graficar_espectro(  x, fs, N  , 'o:', linewidth=1, label='$x_n$')
    graficar_espectro( xx_flattop_f, ff,  fs, 'o--g', linewidth=1)#, label='$x_n\'$ (flattop)')
#    graficar_espectro(xxx_flattop_f, fff, fs, 'x:r', linewidth=1)#, label='$x_n\'\'$ (flattop)')
    
    plt.title('$x_n\'$ y $x_n\'\'$ (Flattop), SNR='+str(int(snr_dB)))
    plt.xlabel('f [Hz]'), plt.ylabel('Amplitud [dB]')
    #plt.legend()
    #plt.show()
    
    #%%
    # Estimadores de a1 y W1
    # a1_est = | F{x(k) . w_i(k)} |
    # W1_est =  arg_max_f{ P_est }
    
    # ------------- a1 --------------- #
    a1_pot_dB = 10.0 * np.log10(np.abs( a1**2/2  )**2)
    
    a1_est_boxcar   = xx_boxcar_f[ff  == W0]
    a1_est_flattop  = xx_flattop_f[ff == W0]
    a1_est_hamming  = xx_hamming_f[ff == W0]
    a1_est_hann     = xx_hann_f[ff == W0]
    
    # Calculo sesgo y varianza
    
    mediana_a1_boxcar   = np.median(a1_est_boxcar)
    mediana_a1_flattop  = np.median(a1_est_flattop)
    mediana_a1_hamming  = np.median(a1_est_hamming)
    mediana_a1_hann     = np.median(a1_est_hann)
    
    sesgo_a1_boxcar   = mediana_a1_boxcar  - a1_pot_dB
    sesgo_a1_flattop  = mediana_a1_flattop - a1_pot_dB
    sesgo_a1_hamming  = mediana_a1_hamming - a1_pot_dB
    sesgo_a1_hann     = mediana_a1_hann    - a1_pot_dB
    
    std_a1_boxcar   = mad_std(a1_est_boxcar)
    std_a1_flattop  = mad_std(a1_est_flattop)
    std_a1_hamming  = mad_std(a1_est_hamming)
    std_a1_hann     = mad_std(a1_est_hann)
    
    plt.figure(10*i+3).clf()
    
    nbins = 80; rang = (-17,0); alpha = 0.7
    plt.hist(a1_est_boxcar.transpose(),  bins=nbins, range=rang, alpha=alpha, label='boxcar')
    plt.hist(a1_est_flattop.transpose(), bins=nbins, range=rang, alpha=alpha, label='flattop')
    plt.hist(a1_est_hamming.transpose(), bins=nbins, range=rang, alpha=alpha, label='hamming')
    plt.hist(a1_est_hann.transpose(),    bins=nbins, range=rang, alpha=alpha, label='hann')
    
    plt.axvline(mediana_a1_boxcar,   linestyle='--', color='grey', label='percentil 50')
    plt.axvline(mediana_a1_flattop,  linestyle='--', color='grey')
    plt.axvline(mediana_a1_hamming,  linestyle='--', color='grey')
    plt.axvline(mediana_a1_hann,     linestyle='--', color='grey')
    plt.axvline(a1_pot_dB,           linestyle='--', color='black', label='real')
    
    plt.title('Histograma para amplitud, SNR='+str(int(snr_dB)))
    plt.xlabel('Amplitud [dB]'), plt.ylabel('n° de muestras')
    plt.legend();
    
    
    #%%
    # ------------- W1 --------------- #
    W1_est_boxcar  = ff[ np.argmax(xx_boxcar_f  [ff <= fs/2], axis=0) ]
    W1_est_flattop = ff[ np.argmax(xx_flattop_f [ff <= fs/2], axis=0) ]
    W1_est_hamming = ff[ np.argmax(xx_hamming_f [ff <= fs/2], axis=0) ]
    W1_est_hann    = ff[ np.argmax(xx_hann_f    [ff <= fs/2], axis=0) ]
    
    # Sesgo y varianza
    
    mediana_W1_boxcar  = np.median(W1_est_boxcar )
    mediana_W1_flattop = np.median(W1_est_flattop)
    mediana_W1_hamming = np.median(W1_est_hamming)
    mediana_W1_hann    = np.median(W1_est_hann   )
    
    sesgo_W1_boxcar  = mediana_W1_boxcar  - np.median(W1)
    sesgo_W1_flattop = mediana_W1_flattop - np.median(W1)
    sesgo_W1_hamming = mediana_W1_hamming - np.median(W1)
    sesgo_W1_hann    = mediana_W1_hann    - np.median(W1)
    
    # Uso el desvío estandar porque el MAD me da cero
    std_W1_boxcar  = np.std(W1_est_boxcar )
    std_W1_flattop = np.std(W1_est_flattop)
    std_W1_hamming = np.std(W1_est_hamming)
    std_W1_hann    = np.std(W1_est_hann   )
    
    nbins = 5; rang = (247.5,252.5); alpha = 0.3
    plt.figure(10*i+4).clf()
    plt.hist(W1_est_boxcar,  bins=nbins, range=rang, alpha=alpha, label='boxcar')
    plt.hist(W1_est_flattop, bins=nbins, range=rang, alpha=alpha, label='flattop')
    plt.hist(W1_est_hamming, bins=nbins, range=rang, alpha=alpha, label='hamming')
    plt.hist(W1_est_hann,    bins=nbins, range=rang, alpha=alpha, label='hann')
    plt.axvline(W0, linestyle='--', color='black', label='real')
    plt.title('Histograma para frecuencia, SNR='+str(int(snr_dB)))
    plt.xlabel('Frecuencia [Hz]'), plt.ylabel('n° de muestras')
    plt.legend();
    
    #%%
    resultados.append([mediana_W1_boxcar,  sesgo_W1_boxcar,  std_W1_boxcar,  
                       mediana_a1_boxcar,  sesgo_a1_boxcar,  std_a1_boxcar ])
    resultados.append([mediana_W1_flattop, sesgo_W1_flattop, std_W1_flattop, 
                       mediana_a1_flattop, sesgo_a1_flattop, std_a1_flattop])
    resultados.append([mediana_W1_hamming, sesgo_W1_hamming, std_W1_hamming, 
                       mediana_a1_hamming, sesgo_a1_hamming, std_a1_hamming])
    resultados.append([mediana_W1_hann,    sesgo_W1_hann,    std_W1_hann,    
                       mediana_a1_hann,    sesgo_a1_hann,    std_a1_hann])
    
#%%

df = DataFrame(resultados, columns=['Mediana $\Omega_1$', 'Sesgo $\Omega_1$', 'Desvío std $\Omega_1$', 
                                    'Mediana $X$', 'Sesgo $X$', 'Desvío std $X$'],
               index=[
                         'Boxcar (3 dB)',
                         'Flattop (3 dB)',
                         'Hamming (3 dB)',
                         'Hann (3 dB)',
                         'Boxcar (10 dB)',
                         'Flattop (10 dB)',
                         'Hamming (10 dB)',
                         'Hann (10 dB)',
                     ])

# Dict used to center the table headers
d = dict(selector="th",
    props=[('text-align', 'center')])

# Style
s = df.style.set_properties(**{'width':'10em', 'text-align':'center'})\
      .set_table_styles([d])
        

HTML(s.to_html())
