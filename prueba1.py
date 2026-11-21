import numpy as np
from src.sistema import respuesta_impulsional, H_analitica

tau = 1e-3          # 1 ms (R = 1 kΩ, C = 1 µF)
dt = tau / 100
t, h = respuesta_impulsional(tau, dt, 10 * tau)

print("h(0) =", h[0], " (esperado 1/tau =", 1 / tau, ")")
print("Área de h ≈", np.sum(h) * dt, " (esperado ≈ 1)")

omega = 1 / tau     # frecuencia de corte
H = H_analitica(omega, tau)
print("|H| =", abs(H), " (esperado 0.707)")
print("fase [grados] =", np.degrees(np.angle(H)), " (esperado -45)")