import numpy as np
import matplotlib.pyplot as plt
from src.entradas import sinusoide_encendida

tau = 1e-3
dt = tau / 100
omega0 = 1 / tau
t_on = 20 * tau

t, x = sinusoide_encendida(1.0, omega0, t_on, dt, 30 * tau)

print("muestras:", len(x))
print("x[0] =", x[0], "(debe ser 0)")
print("maximo de |x| =", np.max(np.abs(x)), "(debe ser casi 1)")
print("maximo despues de t_on =", np.max(np.abs(x[t >= t_on])), "(debe ser 0)")

plt.plot(t * 1e3, x)
plt.xlabel("t [ms]")
plt.ylabel("x [V]")
plt.title("Sinusoide encendida")
plt.show()