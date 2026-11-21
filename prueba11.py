import numpy as np
import matplotlib.pyplot as plt
from src.sistema import respuesta_impulsional, H_analitica

tau = 1e-3
dt = tau / 200
omega = np.logspace(-2, 2, 400) / tau       # de 0.01/tau a 100/tau

H = H_analitica(omega, tau)

print("k   |H_num - H_k|   |H_num - H|")
for k in [3, 5, 10]:
    th, h = respuesta_impulsional(tau, dt, k * tau)
    h[0] = h[0] / 2
    # transformada numerica: suma de rectangulos con el factor dt
    H_num = dt * np.exp(-1j * np.outer(omega, th)) @ h
    H_k = (1 - np.exp(-k - 1j * omega * k * tau)) / (1 + 1j * omega * tau)
    print(k, np.max(np.abs(H_num - H_k)), np.max(np.abs(H_num - H)))

# figura de modulo y fase con las tres frecuencias del proyecto
marcas = np.array([0.2, 1.0, 5.0]) / tau
Hm = H_analitica(marcas, tau)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
ax1.semilogx(omega * tau, 20 * np.log10(np.abs(H)))
ax1.plot(marcas * tau, 20 * np.log10(np.abs(Hm)), "o")
ax1.set_ylabel("|H| [dB]")
ax1.grid(True)

ax2.semilogx(omega * tau, np.degrees(np.angle(H)))
ax2.plot(marcas * tau, np.degrees(np.angle(Hm)), "o")
ax2.set_ylabel("fase [grados]")
ax2.set_xlabel("omega * tau")
ax2.grid(True)

fig.savefig("figura_H.png", dpi=150)
plt.show()