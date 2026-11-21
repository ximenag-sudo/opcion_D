import numpy as np
from src.sistema import respuesta_impulsional
from src.entradas import sinusoide_encendida
from src.convolucion import conv_fft
from src.validacion import salida_analitica

tau = 1e-3
omega0 = 1 / tau
t_on = 20 * tau

print("tau/dt   error normal   error con h[0]/2")
for pasos in [25, 50, 100, 200, 400]:
    dt = tau / pasos
    th, h = respuesta_impulsional(tau, dt, 8 * tau)
    t, x = sinusoide_encendida(1.0, omega0, t_on, dt, 30 * tau)
    y_ana = salida_analitica(1.0, omega0, tau, t_on, t)

    y1 = conv_fft(x, h, dt)[:len(t)]
    h2 = h.copy()
    h2[0] = h2[0] / 2          # regla del trapecio en el salto de h
    y2 = conv_fft(x, h2, dt)[:len(t)]

    print(pasos, np.max(np.abs(y1 - y_ana)), np.max(np.abs(y2 - y_ana)))