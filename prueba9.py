import numpy as np
import pandas as pd
from src.sistema import respuesta_impulsional, H_analitica
from src.entradas import sinusoide_encendida
from src.convolucion import conv_fft
from src.validacion import estimar_amplitud_fase

tau = 1e-3
dt = tau / 200
t_on = 60 * tau

th, h = respuesta_impulsional(tau, dt, 10 * tau)
h[0] = h[0] / 2          # trapecio en el salto de h

filas = []
for factor in [0.2, 1.0, 5.0]:
    omega = factor / tau
    t, x = sinusoide_encendida(1.0, omega, t_on, dt, 80 * tau)
    y = conv_fft(x, h, dt)[:len(t)]
    R, fase = estimar_amplitud_fase(t, y, omega, 10 * tau, 55 * tau)
    H = H_analitica(omega, tau)
    filas.append({
        "omega*tau": factor,
        "|H| teorico": abs(H),
        "amplitud medida": R,
        "fase teorica [gr]": np.degrees(np.angle(H)),
        "fase medida [gr]": np.degrees(fase),
    })

tabla = pd.DataFrame(filas)
tabla["error amplitud"] = (tabla["amplitud medida"] - tabla["|H| teorico"]).abs()
tabla["error fase [gr]"] = (tabla["fase medida [gr]"] - tabla["fase teorica [gr]"]).abs()
print(tabla.to_string(index=False))