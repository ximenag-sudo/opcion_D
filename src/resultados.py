import numpy as np
import pandas as pd
from src.sistema import respuesta_impulsional, H_analitica
from src.entradas import sinusoide_encendida, suma_sinusoides_encendida
from src.convolucion import conv_directa, conv_fft
from src.validacion import salida_analitica, estimar_amplitud_fase, estimar_componentes
TAU = 1e-3


def _h_trapecio(tau, dt, t_max):
    """h muestreada con la mitad del valor en el salto de t = 0."""
    th, h = respuesta_impulsional(tau, dt, t_max)
    h[0] = h[0] / 2
    return th, h


def tabla_metodos(factores=(0.2, 1.0, 5.0), tau=TAU, pasos=100):
    """Directa, FFT y solucion analitica para cada omega*tau."""
    dt = tau / pasos
    t_on = 60 * tau
    th, h = _h_trapecio(tau, dt, 10 * tau)
    filas = []
    for f in factores:
        omega = f / tau
        t, x = sinusoide_encendida(1.0, omega, t_on, dt, 80 * tau)
        yd = conv_directa(x, h, dt)[:len(t)]
        yf = conv_fft(x, h, dt)[:len(t)]
        ya = salida_analitica(1.0, omega, tau, t_on, t)
        filas.append({
            "omega*tau": f,
            "error rel. directa vs FFT": np.linalg.norm(yd - yf) / np.linalg.norm(yd),
            "error max directa vs analitica": np.max(np.abs(yd - ya)),
            "error max FFT vs analitica": np.max(np.abs(yf - ya)),
        })
    return pd.DataFrame(filas)


def tabla_amplitud_fase(factores=(0.2, 1.0, 5.0), tau=TAU, pasos=200):
    """Amplitud y fase medidas en la zona estacionaria contra H(omega)."""
    dt = tau / pasos
    t_on = 60 * tau
    th, h = _h_trapecio(tau, dt, 10 * tau)
    filas = []
    for f in factores:
        omega = f / tau
        t, x = sinusoide_encendida(1.0, omega, t_on, dt, 80 * tau)
        y = conv_fft(x, h, dt)[:len(t)]
        R, fase = estimar_amplitud_fase(t, y, omega, 10 * tau, 55 * tau)
        H = H_analitica(omega, tau)
        filas.append({
            "omega*tau": f,
            "|H| teorico": abs(H),
            "amplitud medida": R,
            "fase teorica [gr]": np.degrees(np.angle(H)),
            "fase medida [gr]": np.degrees(fase),
        })
    tabla = pd.DataFrame(filas)
    tabla["error amplitud"] = (tabla["amplitud medida"] - tabla["|H| teorico"]).abs()
    tabla["error fase [gr]"] = (tabla["fase medida [gr]"] - tabla["fase teorica [gr]"]).abs()
    return tabla


def tabla_truncamiento(ks=(1, 2, 3, 5, 8, 10), tau=TAU, pasos=200):
    """Error de la salida al cortar h en k*tau (omega*tau = 1, antes del apagado)."""
    dt = tau / pasos
    omega = 1 / tau
    t_on = 20 * tau
    t, x = sinusoide_encendida(1.0, omega, t_on, dt, 30 * tau)
    ya = salida_analitica(1.0, omega, tau, t_on, t)
    antes = t < t_on - 5 * dt
    filas = []
    for k in ks:
        th, h = _h_trapecio(tau, dt, k * tau)
        y = conv_fft(x, h, dt)[:len(t)]
        filas.append({
            "k": k,
            "area perdida": 1 - np.sum(h) * dt,
            "error max": np.max(np.abs(y - ya)[antes]),
            "exp(-k)": np.exp(-k),
        })
    return pd.DataFrame(filas)


def tabla_paso(pasos_lista=(25, 50, 100, 200, 400), tau=TAU):
    """Error contra el paso dt, con h[0] completo y con h[0]/2 (antes del apagado)."""
    omega = 1 / tau
    t_on = 20 * tau
    filas = []
    for pasos in pasos_lista:
        dt = tau / pasos
        th, h = respuesta_impulsional(tau, dt, 8 * tau)
        h2 = h.copy()
        h2[0] = h2[0] / 2
        t, x = sinusoide_encendida(1.0, omega, t_on, dt, 30 * tau)
        ya = salida_analitica(1.0, omega, tau, t_on, t)
        antes = t < t_on - 5 * dt
        e1 = np.abs(conv_fft(x, h, dt)[:len(t)] - ya)[antes].max()
        e2 = np.abs(conv_fft(x, h2, dt)[:len(t)] - ya)[antes].max()
        filas.append({"tau/dt": pasos, "error h[0] completo": e1, "error h[0]/2": e2})
    return pd.DataFrame(filas)


def tabla_lineal_vs_circular(tau=TAU, pasos=100):
    """Convolucion circular con distintas longitudes L contra la lineal.

    La senoidal queda encendida hasta el final del registro, asi la cola
    de la respuesta es grande y el efecto de borde se nota.
    """
    dt = tau / pasos
    th, h = _h_trapecio(tau, dt, 8 * tau)
    t, x = sinusoide_encendida(1.0, 1 / tau, 30 * tau, dt, 30 * tau)
    N, M = len(x), len(h)
    y_lin = conv_directa(x, h, dt)
    filas = []
    for L in [N, N + M // 2, N + M - 1]:
        y_c = np.fft.irfft(np.fft.rfft(x, L) * np.fft.rfft(h, L), L) * dt
        dif = y_c - y_lin[:L]
        filas.append({
            "L": L,
            "muestras que dan la vuelta": N + M - 1 - L,
            "error relativo": np.linalg.norm(dif) / np.linalg.norm(y_lin[:L]),
            "diferencia max": np.max(np.abs(dif)),
        })
    return pd.DataFrame(filas)


def error_superposicion(tau=TAU, pasos=200, factores=(0.2, 5.0)):
    """Salida de la suma de senoidales contra la suma de las salidas."""
    dt = tau / pasos
    t_on = 60 * tau
    w1, w2 = factores[0] / tau, factores[1] / tau
    th, h = _h_trapecio(tau, dt, 10 * tau)
    t, x = suma_sinusoides_encendida([1.0, 1.0], [w1, w2], t_on, dt, 80 * tau)
    _, x1 = sinusoide_encendida(1.0, w1, t_on, dt, 80 * tau)
    _, x2 = sinusoide_encendida(1.0, w2, t_on, dt, 80 * tau)
    y = conv_fft(x, h, dt)[:len(t)]
    y1 = conv_fft(x1, h, dt)[:len(t)]
    y2 = conv_fft(x2, h, dt)[:len(t)]
    return np.max(np.abs(y - (y1 + y2)))


def tabla_compuesta(tau=TAU, pasos=200, factores=(0.2, 5.0)):
    """Amplitud y fase de cada componente en la salida de la senoide compuesta."""
    dt = tau / pasos
    t_on = 60 * tau
    omegas = [f / tau for f in factores]
    th, h = _h_trapecio(tau, dt, 10 * tau)
    t, x = suma_sinusoides_encendida([1.0, 1.0], omegas, t_on, dt, 80 * tau)
    y = conv_fft(x, h, dt)[:len(t)]
    medidas = estimar_componentes(t, y, omegas, 10 * tau, 55 * tau)
    filas = []
    for f, w, (R, fase) in zip(factores, omegas, medidas):
        H = H_analitica(w, tau)
        filas.append({
            "omega*tau": f,
            "|H| teorico": abs(H),
            "amplitud medida": R,
            "fase teorica [gr]": np.degrees(np.angle(H)),
            "fase medida [gr]": np.degrees(fase),
        })
    tabla = pd.DataFrame(filas)
    tabla["error amplitud"] = (tabla["amplitud medida"] - tabla["|H| teorico"]).abs()
    tabla["error fase [gr]"] = (tabla["fase medida [gr]"] - tabla["fase teorica [gr]"]).abs()
    return tabla


def tabla_H_truncada(ks=(3, 5, 10), tau=TAU, pasos=200):
    """H analitica, H de h truncada en k*tau (exacta) y transformada numerica."""
    dt = tau / pasos
    omega = np.logspace(-2, 2, 400) / tau
    H = H_analitica(omega, tau)
    filas = []
    for k in ks:
        th, h = _h_trapecio(tau, dt, k * tau)
        H_num = dt * np.exp(-1j * np.outer(omega, th)) @ h
        H_k = (1 - np.exp(-k - 1j * omega * k * tau)) / (1 + 1j * omega * tau)
        filas.append({
            "k": k,
            "|H_num - H_k|": np.max(np.abs(H_num - H_k)),
            "|H_num - H|": np.max(np.abs(H_num - H)),
            "exp(-k)": np.exp(-k),
        })
    return pd.DataFrame(filas)