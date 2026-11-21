import numpy as np
from src.sistema import respuesta_impulsional
from src.entradas import sinusoide_encendida
from src.convolucion import conv_fft
from src.validacion import salida_analitica

tau = 1e-3
dt = tau / 200
omega0 = 1 / tau
t_on = 20 * tau

t, x = sinusoide_encendida(1.0, omega0, t_on, dt, 30 * tau)
y_ana = salida_analitica(1.0, omega0, tau, t_on, t)
antes = t < t_on - 5 * dt

print("k   area perdida   error medido   e^-k/sqrt(2)")
for k in [1, 2, 3, 5, 8, 10]:
    th, h = respuesta_impulsional(tau, dt, k * tau)
    h[0] = h[0] / 2
    y = conv_fft(x, h, dt)[:len(t)]
    error = np.max(np.abs(y - y_ana)[antes])
    area_perdida = 1 - np.sum(h) * dt
    print(k, area_perdida, error, np.exp(-k) / np.sqrt(2))