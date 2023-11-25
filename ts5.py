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
#from astropy.stats import mad_std

def periodogram(x_t, fs, N, k):
    df  = fs/N
    f   = np.linspace(0, (N-1)*df, N)
    spec = np.fft.fft( x_t, axis = 0 )
    X_f = 10 * np.log10(np.abs(k/N * spec)**2)
    return f, X_f

def periodogram_welch(x_t, fs, window):
    N = len(x_t)
    L = N//5
    D = L//2 # 50% overlap
    
    f,Pxx = signal.welch(x_t,
                         fs=fs,
                         window=window,
                         nperseg=L,
                         noverlap=D,
                         nfft=N,
                         average='median',
                         axis=0 
                        )
    Pxx[ 0] = Pxx[ 1]
    Pxx[-1] = Pxx[-2]
    return f, Pxx


# def periodogram_smooth(X, fs, window, M, n1=0, n2=None):
#     N = len(X)
#     delta_f = fs/N
#     X = np.reshape(X, -1)
#     if n2 is None:
#         n2 = N
    
#     R = np.cov(X[n1:n2])
#     r = [np.fliplr(R[0,1:M-1]), R[0,0], R[0,1:M-1]]
#     M = 2*M-1
#     w = np.ones((M,1))
#     match window:
#         case 'hamming':
#             w = signal.windows.hamming(M)
#         case 'hann':
#             w = signal.windows.hann(M)
#         case 'barlett':
#             w = signal.windows.bartlett(M)
#         case 'blackman':
#             w = signal.windows.blackman(M)
#         case _:
#             w = signal.windows.boxcar(M)
    
#     r = np.transpose(r) * w
#     Px = np.abs(np.fft.fft(r, 1024))
#     Px[0] = Px[1]
#     f = np.arange(start = 0, stop = N, step = delta_f)

#     return f, Px

# ver spectrum psdcorrelogram
# w: ventana que va de -M a +M, con M < N-1
def blackman_tukey(x, fs, window, M = None):
    N,R,Z = x.shape
    delta_f = fs/N
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1
    
    w = signal.windows.get_window(window, r_len)
    
    Px = np.zeros_like(x)

    for j in range(Z):
        for i in range(R):
            # hay que aplanar los arrays por np.correlate.
            # usaremos el modo same que simplifica el tratamiento
            # de la autocorr
            xx = x[:,i,j].ravel()[:r_len];
        
            r = np.correlate(xx, xx, mode='same') / r_len
    
            Px[:,i,j] = np.abs(np.fft.fft(r * w, n = N) ) / N

    f  = np.arange(start = 0, stop = N, step = delta_f)
    
    return f, Px;


def graficar_espectro(ax, f, X_f, fs, *args, **kwargs):   
    bfrec = (f <= fs/2)
#    bfrec.resize(max(len(f), len(X_f)), refcheck=False)
    X_f = 20 * np.log10(np.abs(X_f)) # Lo paso a dB
    return ax.plot(f[bfrec], X_f[bfrec], *args, **kwargs);

# def graficar_espectro(x_t, fs, N, k, *args, **kwargs):    
#     df  = fs/N
#     f   = np.linspace(0, (N-1)*df, N)
#     bfrec = (f <= fs/2)
#     X_f = 10 * np.log10(np.abs(k/N * np.fft.fft( x_t, axis = 0 ))**2)

#     plt.plot(f[bfrec], X_f[bfrec], *args, **kwargs)	


def vertical_flaten(x):
    return x.reshape(x.shape[0], 1)

#%%
R = 200  # 200 repeticiones o realizaciones
N = 1000 # nº de muestras

# frec. de sampling normalizada
fs = N
delta_f = fs/N

# Desplazamiento en frecuencia de cada realización
fr = np.random.uniform(-1/2, 1/2, size=(1,R,1))

# Frecuencia central normalizada: pi/2 - es equivalente a fs/4
W0 = fs/4
W1 = W0 + fr

snr_dB = np.array([ 3., 10. ], dtype=np.float64)
var_noise  = 10 ** (snr_dB / (-10)) # varianza (potencia)
noise = np.random.normal(loc=0.0, scale=np.sqrt(var_noise), size=(N,R,2))

a1 = np.sqrt(2) # amplitud normalizada

end = N / fs

n = np.linspace(start=0, stop=end, num = N, endpoint = False).reshape((N,1,1))

x  = a1 * np.sin(2*np.pi * W1 * n) + noise


# Espectros
window='bartlett'
f,  X_welch_f = periodogram_welch(x, fs, window=window)
f2, X_BT_f = blackman_tukey(x, fs, window=window)
#f2, X_BT_f = periodogram_smooth(x, fs, window=window, M=N//5)


#%%
#-------------------------------------------------------------------------------

fig = plt.figure(1)
fig.clf()
axis = fig.subplots(2,1, sharex=True)

for i, ax in enumerate(axis):
    #fig.xticks(np.linspace(0, 500, 5), minor=False)
    ax.set_xticks(np.arange(W0-10, W0+10+1), minor=True)
    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.3)
        
    *line, = graficar_espectro(ax, f, X_welch_f[:,:,i], fs, 'o--g', linewidth=1)
    line[0].set_label('welch')
    
    graficar_espectro(ax, f, np.median(X_welch_f[:,:,i], axis=1), fs, 'b--', 
                      linewidth=1, label='welch median')
    
    *line, = graficar_espectro(ax, f2, X_BT_f[:,:,i], fs, 'x-r', linewidth=1)
    line[0].set_label('blackman tukey')
    
    ax.set_title('Periodogramas (window='+window+', SNR='+str(int(snr_dB[i]))+')')
    ax.set_ylabel('Amplitud [dB]')
    ax.legend()
    ax.set_ylim([-100,  0])

axis[0].set_xlim([W0-20, W0+20])
axis[1].set_xlabel('f [Hz]')
fig.show()

#-------------------------------------------------------------------------------

#%%
# Estimacion de frecuencia W1
W1_est_welch = f[  np.argmax(X_welch_f [f  <= fs/2], axis=0) ]
W1_est_BT    = f2[ np.argmax(X_BT_f    [f2 <= fs/2], axis=0) ]

W1_est_welch_median = np.median(W1_est_welch, axis=0)
W1_est_welch_bias   = W1_est_welch_median - np.median(W1)
W1_est_welch_var    = np.var(W1_est_welch, axis=0)

W1_est_BT_median = np.median(W1_est_BT, axis=0)
W1_est_BT_bias   = W1_est_BT_median - np.median(W1)
W1_est_BT_var    = np.var(W1_est_BT, axis=0)


nbins = 5; rang = (W0-2.5,W0+2.5); alpha = 0.5

fig = plt.figure(2)
fig.clf()
axis = fig.subplots(2,1, sharex=True)

for i, ax in enumerate(axis):
    ax.hist(W1_est_welch[:,i], bins=nbins, range=rang, alpha=alpha, label='welch', align='mid')
    ax.hist(W1_est_BT[:,i],    bins=nbins, range=rang, alpha=alpha, label='BT',    align='mid')
    ax.axvline(W0, linestyle='--', color='black', label='real')
    ax.set_title('Histograma para frecuencia, SNR='+str(int(snr_dB[i])))
    ax.set_ylabel('n° de muestras')
    ax.legend();

ax.set_xlabel('Frecuencia [Hz]')
#fig.set_figheight(6)
#fig.set_figwidth(10)
fig.show()

#%%

variability_welch_est = W1_est_welch_var / (W1_est_welch_median**2)
variability_BT_est = W1_est_BT_var / (W1_est_BT_median**2)

variability = np.array((variability_welch_est, variability_BT_est)).T

# L = N//5
# D = L//2
# K = 2*N/L-1
# variability_welch = 9/(8*K)


df = DataFrame(variability, columns=['$i_{W}$', '$i_{BT}$'],
                index=[
                          '3 dB',
                          '10 dB',
                      ])

# # Dict used to center the table headers
# d = dict(selector="th",
#     props=[('text-align', 'center')])

# # Style
# s = df.style.set_properties(**{'width':'10em', 'text-align':'center'})\
#       .set_table_styles([d])
        
HTML(df.to_html(classes="table",justify="center"))