import numpy as np
from src.sistema import respuesta_impulsional
from src.entradas import sinusoide_encendida
from src.convolucion import conv_fft
from src.validacion import salida_analitica

tau = 1e-3
omega0 = 1 / tau
t_on = 20 * tau

print("tau/dt  normal(antes)  trapecio(antes)  t del max trapecio [ms]")
for pasos in [25, 50, 100, 200, 400]:
    dt = tau / pasos
    th, h = respuesta_impulsional(tau, dt, 8 * tau)
    t, x = sinusoide_encendida(1.0, omega0, t_on, dt, 30 * tau)
    y_ana = salida_analitica(1.0, omega0, tau, t_on, t)

    h2 = h.copy()
    h2[0] = h2[0] / 2
    e1 = np.abs(conv_fft(x, h, dt)[:len(t)] - y_ana)
    e2 = np.abs(conv_fft(x, h2, dt)[:len(t)] - y_ana)

    antes = t < t_on - 5 * dt      # zona sin influencia del apagado
    print(pasos, e1[antes].max(), e2[antes].max(), t[np.argmax(e2)] * 1e3)