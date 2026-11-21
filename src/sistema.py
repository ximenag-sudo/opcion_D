import numpy as np


def respuesta_impulsional(tau, dt, t_max):
    """Respuesta impulsional del filtro RC pasabajos.

    h(t) = (1/tau) * exp(-t/tau) para t >= 0.

    Argumentos:
        tau:   constante de tiempo RC [s], debe ser > 0.
        dt:    paso temporal [s], debe ser > 0.
        t_max: duración total muestreada de h [s], debe ser > 0.

    Retorna:
        t: arreglo de tiempos [s]
        h: valores de h(t) [1/s]
    """
    if tau <= 0 or dt <= 0 or t_max <= 0:
        raise ValueError("tau, dt y t_max deben ser positivos")
    t = np.arange(0, t_max, dt)
    h = np.exp(-t / tau) / tau
    return t, h


def H_analitica(omega, tau):
    """Función de transferencia H(w) = 1 / (1 + i*w*tau). Adimensional."""
    return 1.0 / (1.0 + 1j * omega * tau)