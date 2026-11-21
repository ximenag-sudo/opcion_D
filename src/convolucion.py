import numpy as np


def conv_directa(x, h, dt):
    """Convolucion lineal y[n] = dt * suma_k h[k] * x[n-k].

    Devuelve N + M - 1 muestras. dt en segundos.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("x y h deben ser arreglos de una dimension")
    if dt <= 0:
        raise ValueError("dt debe ser positivo")

    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)

    # cada h[k] aporta h[k]*x, empezando en la posicion k de la salida
    for k in range(M):
        y[k:k + N] += h[k] * x

    return y * dt


def eje_salida(dt, N, M, t0=0.0):
    """Tiempos de la salida (N + M - 1 puntos). t0 = suma de los inicios de x y h."""
    return t0 + dt * np.arange(N + M - 1)
def conv_fft(x, h, dt):
    """Convolucion lineal por FFT, rellenando con ceros hasta N + M - 1."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("x y h deben ser arreglos de una dimension")
    if dt <= 0:
        raise ValueError("dt debe ser positivo")

    L = len(x) + len(h) - 1
    X = np.fft.rfft(x, L)    # el segundo argumento rellena con ceros hasta L
    H = np.fft.rfft(h, L)
    y = np.fft.irfft(X * H, L)
    return y * dt