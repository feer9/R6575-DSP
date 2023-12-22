#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:53:01 2023

@author: feer
"""
import numpy  as np
from scipy import signal as sig
from scipy import io as sio
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

# 4) En el archivo ECG_TP4.mat encontrará un registro electrocardiográfico (ECG)
# registrado durante una prueba de esfuerzo, junto con una serie de variables
# descriptas a continuación. Diseñe y aplique los filtros digitales necesarios
# para mitigar las siguientes fuentes de contaminación:
#
#  - Ruido causado por el movimiento de los electrodos (Alta frecuencia)
#  - Ruido muscular (Alta frecuencia)
#  - Movimiento de la línea de base del ECG, inducido en parte por la 
#  respiración (Baja frecuencia)

data = sio.loadmat('ECG_TP4.mat')
# ecg_lead: Registro de ECG muestreado a fs=1 KHz durante una prueba de esfuerzo
# qrs_pattern1: Complejo de ondas QRS normal
# heartbeat_pattern1: Latido normal
# heartbeat_pattern2: Latido de origen ventricular
# qrs_detections: vector con las localizaciones (en # de muestras) donde ocurren 
#                 los latidos


def periodogram_welch(x_t, fs, window, nfft=None, L=None, D=None):
    N = len(x_t)
    if nfft is None:
        nfft = N
    if L is None:
        L = N//10
    if D is None:
        D = L//2 # 50% overlap

    f,Pxx = sig.welch(x_t,
                      fs=fs,
                      window=window,
                      nperseg=L,
                      noverlap=D,
                      nfft=nfft,
                      average='median',
                      axis=0 
                     )
    Pxx[ 0] = Pxx[ 1]
    Pxx[-1] = Pxx[-2]
    return f, Pxx

def graficar_espectro(f, X_f, fs, *args, **kwargs):   
    bfrec = (f <= fs/2)
    X_f = 20 * np.log10(np.abs(X_f)) # Lo paso a dB
    plt.plot(f[bfrec], X_f[bfrec], *args, **kwargs)
    
def vertical_flaten(x):
    return x.reshape(x.shape[0], 1)


ecg_lead = data['ecg_lead']
qrs_detections = data['qrs_detections']
detections_len = len(qrs_detections)

fs = 1000
Ts = 1/fs

#%%
muestra_inicial = ecg_lead[9837:24015]

N = len(muestra_inicial)
t = np.arange(start = 0, stop = N*Ts, step = Ts)

detections_muestra_inicial = qrs_detections[
    np.logical_and(qrs_detections>9837, qrs_detections<24015)] - 9837

detections_inicial_len = len(detections_muestra_inicial)

plt.figure(1).clf()
plt.plot(t, muestra_inicial)

plt.plot(t[detections_muestra_inicial], muestra_inicial[detections_muestra_inicial], 'o')

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Muestra inicial latidos normales')

#%%
window='blackman'

f, spec_inicial = periodogram_welch(muestra_inicial, fs, window)

plt.figure(2).clf()
graficar_espectro(f, spec_inicial, fs, label='espectro base')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('X [dB]')
plt.title('Espectro muestra inicial');


#%%

# 270 para atras y 330 para adelante
wleft = 220; wright = 380; w = wleft+wright
nmuestras = np.zeros((wleft+wright, detections_inicial_len))


for i in range(detections_inicial_len):
    x = detections_muestra_inicial[i]
    nmuestras[:,i] = muestra_inicial[x-wleft:x+wright].reshape(w,)
    nmuestras[:,i] -= np.median(nmuestras[:,i])
#    sig.detrend(nmuestras[:,i], axis=0, overwrite_data=True)



plt.figure(3).clf()
plt.plot(nmuestras, 'g');
plt.plot(np.mean(nmuestras,axis=1), 'r')
plt.title('Latidos muestra inicial (normales)');

#%%

f, spec_muestras_inicial = periodogram_welch(nmuestras, fs, window)
spec_mean = np.mean(spec_muestras_inicial, axis=1)

plt.figure(2)
graficar_espectro(f, spec_mean, fs, label='media muestral de Welch')
plt.legend();


#%%
# ver hasta q frecuencia tenemos el 90% de potencia para determinar el BW
# podemos usar np.cumsum()

total = np.sum(np.abs(spec_mean))
threshold = 0.98 * total
val = 0

for y in reversed(range(int(np.max(spec_mean)))):
    if spec_mean[(spec_mean > y)].sum() >= threshold:
        val = y
        break

BW = f[(spec_mean>val)]
fci = BW.min()
fcs = BW.max()

print("fci muestra inicial = " + str(fci))
print("fcs muestra inicial = " + str(fcs))
print("")
plt.axvline(fci, linestyle='--', color='black', label='fci')
plt.axvline(fcs, linestyle='--', color='black', label='fcs');

#%%






















# Ahora voy con la señal completa

N = len(ecg_lead)
t = np.arange(start = 0, stop = N*Ts, step = Ts)
td = t[qrs_detections] # vector tiempo de detecciones

plt.figure(5).clf()
plt.plot(t, ecg_lead)

# plt.plot(qrs_detections, ecg_lead[qrs_detections].reshape((detections_len,1)), 
#          'o', label='detections')
plt.title('Muestra completa ECG')
#%%

f, spec = periodogram_welch(ecg_lead, fs, window)

plt.figure(6).clf()
graficar_espectro(f, spec, fs, label='espectro base')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('X [dB]')
plt.title('Espectro muestra completa ECG');

#%%

wleft = 250; wright = 350; w = wleft+wright
nmuestras = np.zeros((w, detections_len))

qrs_detections = qrs_detections.reshape(qrs_detections.shape[0],)
ecg_lead = ecg_lead.reshape(ecg_lead.shape[0],)

# Separo las muestras
for i in range(detections_len):
    x = qrs_detections[i]
    nmuestras[:,i] = ecg_lead[x-wleft : x+wright]
    sig.detrend(nmuestras[:,i], axis=0, overwrite_data=True)

### separación visual, no muy reliable pero funciona
ventriculares = nmuestras[218,:] > 1610  
nmuestras_ventric = nmuestras[:,ventriculares]
nmuestras_normales = nmuestras[:,~ventriculares]

plt.figure(7).clf()
ax = plt.gca()
*line, = ax.plot(nmuestras_ventric, 'b')
line[0].set_label('ventriculares')
*line, = ax.plot(nmuestras_normales, 'g')
line[0].set_label('normales')
plt.title('Latidos muestra completa agrupados')
plt.legend();

plt.figure(8).clf()
plt.plot(np.mean(nmuestras_ventric,  axis=1), 'b', label='media ventriculares')
plt.plot(np.mean(nmuestras_normales, axis=1), 'g', label='media normales')
plt.title('Latidos promedio')
plt.legend();

plt.figure(5)
plt.plot(td[ventriculares], ecg_lead[qrs_detections][ventriculares], 
         '^', label='ventriculares')
plt.plot(td[~ventriculares], ecg_lead[qrs_detections][~ventriculares], 
         'v', label='normales')
plt.legend()
#%%

f, spec_ecg = periodogram_welch(nmuestras, fs, window, nfft=2*w)
spec_mean = np.mean(spec_ecg, axis=1)

plt.figure(6)
graficar_espectro(f, spec_mean, fs, label='media muestral de Welch')

#%%

total = np.sum(np.abs(spec_mean))
threshold = 0.90 * total
val = 0

# Calculo el ancho de banda
for y in reversed(range(int(np.max(spec_mean)))):
    if spec_mean[(spec_mean > y)].sum() >= threshold:
        val = y
        break

BW = f[(spec_mean>val)]
fci = BW.min()
fcs = BW.max()

print("fci total = " + str(fci))
print("fcs total = " + str(fcs))
plt.axvline(fci, linestyle='--', color='black', label='fci')
plt.axvline(fcs, linestyle='--', color='black', label='fcs');


#%%

# filter design
ripple = 1 / 2 # dB
atenuacion = 40 / 2 # dB

ws1 = fci+0.1 #Hz
wp1 = fci+1   #Hz
wp2 = fcs-5   #Hz
ws2 = fcs+5   #Hz

nyq_frec = fs/2

frecs = np.array([0.0, ws1, wp1, wp2, ws2, nyq_frec]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)


bp_sos_butter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, 
                              ws=np.array([ws1, ws2]) / nyq_frec, 
                              gpass=0.5, 
                              gstop=40., 
                              analog=False, 
                              ftype='butter', 
                              output='sos')

# Filtrado
ecg_filt = sig.sosfiltfilt(bp_sos_butter, ecg_lead)


plt.figure(9).clf()
plt.plot(t, ecg_filt)
plt.plot(td[ventriculares], ecg_filt[qrs_detections][ventriculares], 
         '^', label='ventriculares')
plt.plot(td[~ventriculares], ecg_filt[qrs_detections][~ventriculares], 
         'v', label='normales')
plt.title('ECG Filtrado')
plt.legend()

f, spec_filt = periodogram_welch(ecg_filt, fs, window) 

plt.figure(6)
graficar_espectro(f, spec_filt, fs, label='espectro filtrado')
plt.ylim([-100, 125])


nmuestras_filt = np.zeros((w, detections_len))

# Separo las muestras filtradas
for i in range(detections_len):
    x = qrs_detections[i]
    nmuestras_filt[:,i] = ecg_filt[x-wleft : x+wright]
    sig.detrend(nmuestras_filt[:,i], axis=0, overwrite_data=True)

# Espectro filtrado, media por muestras
f, spec_ecg_filt = periodogram_welch(nmuestras_filt, fs, window, nfft=2*w)
spec_mean_filt = np.mean(spec_ecg_filt, axis=1)


graficar_espectro(f, spec_mean_filt, fs, '--', label='media muestral de Welch filtrada')
plt.legend();

# filtros FIR mejor hacerlos con FDATool de Matlab
#%%

nmuestras_ventric_f  = nmuestras_filt[:,ventriculares]
nmuestras_normales_f = nmuestras_filt[:,~ventriculares]

plt.figure(10).clf()
ax = plt.gca()
*line, = ax.plot(nmuestras_ventric_f, 'b')
line[0].set_label('ventriculares')
*line, = ax.plot(nmuestras_normales_f, 'g')
line[0].set_label('normales')
plt.title('Latidos muestra completa filtrados, sin agrupar')
plt.legend();

#%%
# Filtrado no lineal

d1 = qrs_detections-80
cs_y = ecg_lead[d1]
cs_x = t[d1]
cs = CubicSpline(cs_x, cs_y)

#%%
# Filtro de mediana

med200 = sig.medfilt(ecg_lead, kernel_size=201)
med600 = sig.medfilt(med200, kernel_size=601)

#%%
plt.figure(11).clf()
plt.plot(ecg_lead, ':', label='Original')
plt.plot(td*fs, ecg_lead[qrs_detections], 
         'x', label='detections')
plt.plot(cs_x*fs, cs_y, 'o', label='Pt Ref')
plt.plot(cs(t), label='Cubic Spline')
plt.legend(loc='lower center')
plt.title('Filtro CubicSpline - Línea de base')

#%%
fig = plt.figure(12)
fig.clf()
(ax1,ax2) = fig.subplots(2,1, sharex=True)

ax1.plot(ecg_lead, ':', linewidth=0.8, label='Original')
ax1.plot(med200, '--', label='median 200')
ax1.plot(med600, label='median 600')
ax1.plot(cs(t),  label='cubic spline')
ax1.legend(loc='lower left')
ax1.set_title('Filtros no lineales - Línea de base')

ax2.plot(ecg_lead, ':', linewidth=0.8, label='Original')
ax2.plot([0])
ax2.plot(ecg_lead-med600, label='Filtrada (mediana)')
ax2.plot(ecg_lead-cs(t), label='Filtrada (CSpline)')
ax2.legend(loc='lower left')
ax2.set_title('Filtros no lineales - Comparativa')

#%%
# Comparativa filtros lineales vs no lineales
plt.figure(13).clf()

plt.plot(ecg_lead, ':', linewidth=0.8, label='Original')
plt.plot(ecg_filt, label='IIR')
plt.plot(ecg_lead-med600, label='Mediana')
plt.plot(ecg_lead-cs(t), label='CSpline')
plt.legend(loc='lower left')
plt.title('Filtros lineales y no lineales')

