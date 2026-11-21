import numpy as np
import matplotlib.pyplot as plt
from src.sistema import respuesta_impulsional
from src.entradas import sinusoide_encendida
from src.convolucion import conv_directa, conv_fft
from src.validacion import salida_analitica

tau = 1e-3
dt = tau / 100
omega0 = 1 / tau
t_on = 20 * tau

th, h = respuesta_impulsional(tau, dt, 8 * tau)
t, x = sinusoide_encendida(1.0, omega0, t_on, dt, 30 * tau)

y_dir = conv_directa(x, h, dt)[:len(t)]
y_fft = conv_fft(x, h, dt)[:len(t)]
y_ana = salida_analitica(1.0, omega0, tau, t_on, t)

print("directa vs analitica, error max:", np.max(np.abs(y_dir - y_ana)))
print("FFT vs analitica, error max:", np.max(np.abs(y_fft - y_ana)))

plt.plot(t * 1e3, y_ana, label="analitica")
plt.plot(t * 1e3, y_dir, "--", label="convolucion directa")
plt.xlabel("t [ms]")
plt.ylabel("V_C [V]")
plt.legend()
plt.show()