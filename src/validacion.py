import numpy as np


def salida_analitica(A, omega0, tau, t_on, t):
    """Voltaje del capacitor para x = A*sin(omega0*t) entre 0 y t_on. Tiempos en s."""
    modulo = A / np.sqrt(1 + (omega0 * tau) ** 2)
    theta = -np.arctan(omega0 * tau)

    def encendida(tt):
        return modulo * (np.sin(omega0 * tt + theta) - np.sin(theta) * np.exp(-tt / tau))

    y = np.where(t < t_on, encendida(t), 0.0)
    y_apagado = encendida(t_on)
    despues = t >= t_on
    y[despues] = y_apagado * np.exp(-(t[despues] - t_on) / tau)
    return y


def estimar_amplitud_fase(t, y, omega, t_ini, t_fin):
    """Ajusta y ~ a*sin(wt) + b*cos(wt) entre t_ini y t_fin (minimos cuadrados).

    Devuelve amplitud [V] y fase [rad] respecto a sin(wt).
    """
    ventana = (t >= t_ini) & (t <= t_fin)
    if ventana.sum() < 10:
        raise ValueError("ventana demasiado corta")
    tt = t[ventana]
    base = np.column_stack([np.sin(omega * tt), np.cos(omega * tt)])
    (a, b), *_ = np.linalg.lstsq(base, y[ventana], rcond=None)
    return np.hypot(a, b), np.arctan2(b, a)


def estimar_componentes(t, y, omegas, t_ini, t_fin):
    """Ajuste simultaneo de varias sinusoides entre t_ini y t_fin.

    Devuelve una lista de (amplitud [V], fase [rad]), una por cada omega.
    """
    ventana = (t >= t_ini) & (t <= t_fin)
    tt = t[ventana]
    columnas = []
    for w in omegas:
        columnas += [np.sin(w * tt), np.cos(w * tt)]
    coef, *_ = np.linalg.lstsq(np.column_stack(columnas), y[ventana], rcond=None)
    return [(np.hypot(coef[2 * i], coef[2 * i + 1]),
             np.arctan2(coef[2 * i + 1], coef[2 * i])) for i in range(len(omegas))]