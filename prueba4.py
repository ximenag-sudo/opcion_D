import numpy as np
from src.sistema import respuesta_impulsional
from src.entradas import sinusoide_encendida
from src.convolucion import conv_directa, conv_fft

tau = 1e-3
dt = tau / 100
th, h = respuesta_impulsional(tau, dt, 8 * tau)

# sinusoide que sigue encendida hasta el final del registro
t, x = sinusoide_encendida(1.0, 1 / tau, 30 * tau, dt, 30 * tau)

y_dir = conv_directa(x, h, dt)
y_fft = conv_fft(x, h, dt)

err = np.linalg.norm(y_dir - y_fft) / np.linalg.norm(y_dir)
print("error relativo directa vs FFT:", err)

# caso sin relleno suficiente: FFT de longitud N en vez de N + M - 1
N = len(x)
y_circ = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(h, N), N) * dt
err_circ = np.linalg.norm(y_circ - y_dir[:N]) / np.linalg.norm(y_dir[:N])
print("error relativo sin relleno suficiente:", err_circ)
print("diferencia en las primeras muestras:", np.max(np.abs(y_circ[:50] - y_dir[:50])))