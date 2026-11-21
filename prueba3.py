import numpy as np
from src.sistema import respuesta_impulsional
from src.entradas import sinusoide_encendida
from src.convolucion import conv_directa, eje_salida

# 1. ejemplo hecho a mano: x=[2,3,1], h=[1,0.5], dt=1 -> [2,4,2.5,0.5]
y = conv_directa([2, 3, 1], [1, 0.5], 1.0)
print("ejemplo a mano:", y, "(esperado [2, 4, 2.5, 0.5])")

# 2. escalon contra la carga del capacitor: 1 - exp(-t/tau)
tau = 1e-3
dt = tau / 100
th, h = respuesta_impulsional(tau, dt, 8 * tau)
x = np.ones(len(th))
y = conv_directa(x, h, dt)
ty = eje_salida(dt, len(x), len(h))
n = len(th)  # hasta aqui h ya cubre todo el pasado
error = np.max(np.abs(y[:n] - (1 - np.exp(-ty[:n] / tau))))
print("error maximo en el escalon:", error, "(del orden de dt/tau = 0.01)")

# 3. comprobacion extra con numpy.convolve, solo para comparar
t, xs = sinusoide_encendida(1.0, 1 / tau, 20 * tau, dt, 30 * tau)
y1 = conv_directa(xs, h, dt)
y2 = np.convolve(xs, h) * dt
print("diferencia con np.convolve:", np.max(np.abs(y1 - y2)))