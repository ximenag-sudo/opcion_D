import numpy as np
import pandas as pd
from src.sistema import respuesta_impulsional, H_analitica
from src.entradas import sinusoide_encendida, suma_sinusoides_encendida
from src.convolucion import conv_fft
from src.validacion import estimar_componentes

tau = 1e-3
dt = tau / 200
t_on = 60 * tau
w1, w2 = 0.2 / tau, 5.0 / tau

th, h = respuesta_impulsional(tau, dt, 10 * tau)
h[0] = h[0] / 2

# señal compuesta y su salida
t, x = suma_sinusoides_encendida([1.0, 1.0], [w1, w2], t_on, dt, 80 * tau)
y = conv_fft(x, h, dt)[:len(t)]

# superposicion: salida de la suma contra suma de las salidas
_, x1 = sinusoide_encendida(1.0, w1, t_on, dt, 80 * tau)
_, x2 = sinusoide_encendida(1.0, w2, t_on, dt, 80 * tau)
y1 = conv_fft(x1, h, dt)[:len(t)]
y2 = conv_fft(x2, h, dt)[:len(t)]
print("superposicion, error max:", np.max(np.abs(y - (y1 + y2))))

# amplitud y fase de cada componente en la salida compuesta
medidas = estimar_componentes(t, y, [w1, w2], 10 * tau, 55 * tau)
filas = []
for w, (R, fase) in zip([w1, w2], medidas):
    H = H_analitica(w, tau)
    filas.append({
        "omega*tau": w * tau,
        "|H| teorico": abs(H),
        "amplitud medida": R,
        "fase teorica [gr]": np.degrees(np.angle(H)),
        "fase medida [gr]": np.degrees(fase),
    })
tabla = pd.DataFrame(filas)
tabla["error amplitud"] = (tabla["amplitud medida"] - tabla["|H| teorico"]).abs()
tabla["error fase [gr]"] = (tabla["fase medida [gr]"] - tabla["fase teorica [gr]"]).abs()
print(tabla.to_string(index=False))