#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 19:25:09 2023

@author: feer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    end = nn / fs
    tt  = np.linspace(start=0, stop=end, num = nn, endpoint = False)
    xx = vmax * np.sin(2. * np.pi * ff * tt + ph) + dc
    return tt,xx


def graficar_espectro(f, X_f, fs, label='', *args):
	bfrec = f <= fs/2
#	plt.clf()
	plt.plot(f[bfrec], 10 * np.log10(2*np.abs(X_f[bfrec])**2), label=label)
#	plt.xlabel('f [Hz]'), plt.ylabel('Amplitud [dB]')

def graficar_tiempo(t, x_t, label=''):
	plt.clf()
	plt.plot(t, x_t, label=label)
	plt.xlabel('t [s]'), plt.ylabel('Amplitud [V]')

#%%

# Parámetros de mi señal
fx = 50 #fs/N # Hz
Ax = 1.0 * np.sqrt(2) # Volts - la amplitud sqrt(2) me dará una potencia unitaria
offset = 0 # Volts
phase = 0

#Parámetros internos del ADC
os=10 # oversampling
ADC_fs = 200
ADC_N  = ADC_fs

# Parámetros de la simulación
fs = ADC_fs * os # Hz
N  = fs  # n° de muestras
Ts = 1 / fs # segundos
T_simulacion = N * Ts # segundos


#%%
# Genero mi señal senoidal con los parámetros anteriores
t, x_t  = mi_funcion_sen(Ax, offset, fx, phase, N, fs)

SNR_dB = 20 # signal to noise ratio, valor deseado
variance_noise  = 10 ** (SNR_dB / (-10))

# Ruido de amplificacion: ruido analogico, distribucion normal
noise_analog = np.random.normal(loc=0.0, scale=np.sqrt(variance_noise), size=N)
s_R = x_t + noise_analog

# ruido de cuantizacion es el uniforme.

#%% ADC
# Visualice en una misma gráfica sR y sQ, donde se pueda observar que tienen
# el mismo rango en Volts y el efecto de la cuantización para VF = 2 Volts 
# y  B = 4, 8 y 16 bits.

B_bits = 4
#B_bits_n = 2**B_bits
Vf = 2
q = (2*Vf) / (2**B_bits - 1)

#noise_cuantizacion_esperado = np.random.uniform(-q/2, q/2, N)

Vmax = np.max(s_R)
Vmin = np.min(s_R)

Vpp = Vmax-Vmin

# Valor de escala deseado: 90% de Vf
k_load = 0.9
k = k_load * (2*Vf) / Vpp

# Escalo mi señal a +- 90% de Vf 
s_R *= k
# Saturo a ±Vf
np.clip(s_R, -Vf, Vf, out=s_R)

#%% Filtro la señal analógica
ftran = 0.1
fstop = np.min([1/os + ftran/2, 1/os * 5/4])
fpass = np.max([fstop - ftran/2, fstop * 3/4])
ripple = 0.5 # dB
attenuation = 40 # dB

# como usaremos filtrado bidireccional, alteramos las restricciones para
# ambas pasadas
ripple = ripple / 2 # dB
attenuation = attenuation / 2 # dB

orderz, wcutofz = sig.buttord( fpass, fstop, ripple, attenuation, analog=False)

filter_sos = sig.iirfilter(orderz, wcutofz, rp=ripple, rs=attenuation, \
                            btype='lowpass', \
                            analog=False, \
                            ftype='butter', \
                            output='sos')

#filter_sos = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] # sin filtro

# Filtrado
s_R  = sig.sosfiltfilt(filter_sos, s_R)

#%% Cuantización de la señal

s_Q = np.floor(s_R / q) * q + q/2

# Señal de ruido (noise de quantización)
nq = s_Q - s_R
#np.mean(nq)

#%% Muestreo (diezmado)

ADC_t    = t[::os]
ADC_s_Q = s_Q[::os]


#%% Análisis espectral

# Obtengo el espectro de las señales
df  = fs/N
f   = np.linspace(0, (N-1)*df, N)
ADC_df = ADC_fs/ADC_N
ADC_f  = np.linspace(0, (ADC_N-1)*ADC_df, ADC_N)

# Señal en el tiempo
X_f = 1/N * np.fft.fft( s_R, axis = 0 )

# Cuantización
s_Q_f = 1/N * np.fft.fft( s_Q, axis = 0 )

# Muestreo diezmado
ADC_s_Q_f = 1/ADC_N * np.fft.fft( ADC_s_Q, axis = 0 )



#%%

fig1 = plt.figure(1)
fig1.clf()
ax = fig1.add_subplot()
plt.title('Señal de ' + str(fx) + ' Hz en el tiempo')
plt.xlabel('t [s]'), plt.ylabel('Amplitud [V]')
yticks_major_q = q
while yticks_major_q < Vf/8: 
	yticks_major_q *= 2
ax.set_yticks(np.arange(-Vf, Vf, yticks_major_q))
if B_bits < 8:
	ax.set_yticks(np.arange(-Vf, Vf+q, q), minor=True)
ax.set_xticks(np.linspace(0, 1/fx, fs//fx//os+1))
ax.set_xticks(np.linspace(0, 1/fx, fs//fx+1), minor=True)
ax.set_xlim([0, 1/fx]) #T_simulacion])
ax.set_ylim([-Vf, Vf])
ax.grid(which='major', alpha=0.7)
ax.grid(which='minor', alpha=0.3)

ax.plot(t, s_R, label='$s_{R}$ (señal en tiempo)')
ax.plot(t,    s_Q,'o', label='$s_{Q}$ (cuantización)')
ax.plot(ADC_t, ADC_s_Q,'x', label='$s_{Q (D)}$ (muestreo diezmado)', markersize=18)
ax.plot(t, nq, label='$e = s_Q - s_R$ (error de cuantización)')
          # nq/np.max(nq)*Vf/8

ax.legend()
plt.show()


plt.figure(2)
plt.clf()
plt.title('Espectro señal ' + str(fx) + ' Hz con SNR = ' + str(SNR_dB) + ' dB')
plt.xlabel('f [Hz]'), plt.ylabel('Amplitud [dB]')
plt.xlim([0, fs/2])
graficar_espectro(f, X_f, fs, label='señal + ruido filtrado')
graficar_espectro(f, s_Q_f, fs, label='señal cuantizada')
graficar_espectro(ADC_f, ADC_s_Q_f, ADC_fs, label='cuantizada y muestreada')
plt.legend()
plt.grid()
plt.arrow(x=fx+82, y=0, dx=-60, dy=-1, width=3, head_width=8, head_length=20)
plt.show()


#%% Verificaciones

# Densidad espectral de potencia 
DEP_X_f_dB = 10 * np.log10(2*np.abs(X_f)**2)

# Valor medio de densidad epectral de potencia
DEP_X_f_mean = np.mean(DEP_X_f_dB)

# tengo el SNR_dB=10 + los 30 por la longitud del vector
# valor medio ≈ -10 log10(sigmacuadrado) - 10 log10(N)

# la SNR es varianza por N

variance_noise_est = np.var(noise_analog)
#variance_s_R = np.var(s_R)

SNR_dB_noise_estimado = 10 * np.log10(variance_noise_est) - 10 * np.log10(N)


variance_noise_q = q**2 / 12

# Idealmente debería ser cercano a 1
relacion_rudio_an_dig = variance_noise_est / variance_noise_q

#   Bonus:
# Analizar la señal de error e=sQ−sR verificando las descripciones estadísti-
# cas vistas en teoría (Distribución uniforme, media, varianza, incorrelación)
