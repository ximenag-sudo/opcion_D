import numpy as np


def sinusoide_encendida(A, omega0, t_on, dt, t_total):
    """x(t) = A*sin(omega0*t) entre 0 y t_on, y cero fuera. Tiempos en s, A en V."""
    if A <= 0 or omega0 <= 0 or t_on <= 0 or dt <= 0:
        raise ValueError("A, omega0, t_on y dt deben ser positivos")
    if t_total < t_on:
        raise ValueError("t_total no puede ser menor que t_on")

    t = np.arange(0, t_total, dt)
    x = np.where(t < t_on, A * np.sin(omega0 * t), 0.0)
    return t, x


def suma_sinusoides_encendida(amplitudes, omegas, t_on, dt, t_total):
    """Suma de sinusoides A_i*sin(w_i*t) entre 0 y t_on, cero fuera. Tiempos en s."""
    if len(amplitudes) != len(omegas):
        raise ValueError("amplitudes y omegas deben tener la misma longitud")
    t = np.arange(0, t_total, dt)
    x = np.zeros_like(t)
    for A, w in zip(amplitudes, omegas):
        x += A * np.sin(w * t)
    x[t >= t_on] = 0.0
    return t, x