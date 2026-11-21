import numpy as np
import matplotlib.pyplot as plt
from src.sistema import respuesta_impulsional, H_analitica
from src.entradas import sinusoide_encendida, suma_sinusoides_encendida
from src.convolucion import conv_directa, conv_fft
from src.validacion import salida_analitica

TAU = 1e-3


def _h(tau, dt, t_max):
    """h muestreada con la mitad del valor en el salto de t = 0."""
    th, h = respuesta_impulsional(tau, dt, t_max)
    h[0] = h[0] / 2
    return th, h


def figura_H(tau=TAU, factores=(0.2, 1.0, 5.0), guardar=None):
    """Modulo (dB) y fase (grados) de H(w). Marca las frecuencias omega*tau dadas."""
    omega = np.logspace(-2, 2, 400) / tau
    H = H_analitica(omega, tau)
    marcas = np.array(factores) / tau
    Hm = H_analitica(marcas, tau)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    ax1.semilogx(omega * tau, 20 * np.log10(np.abs(H)))
    ax1.plot(marcas * tau, 20 * np.log10(np.abs(Hm)), "o")
    ax1.set_ylabel("|H| [dB]")
    ax1.set_title(r"$H(\omega)=1/(1+i\omega\tau)$")
    ax1.grid(True)

    ax2.semilogx(omega * tau, np.degrees(np.angle(H)))
    ax2.plot(marcas * tau, np.degrees(np.angle(Hm)), "o")
    ax2.set_ylabel("fase [grados]")
    ax2.set_xlabel(r"$\omega\tau$")
    ax2.grid(True)

    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def figura_compuesta(tau=TAU, factores=(0.2, 5.0), pasos=100, guardar=None):
    """Entrada de dos senoidales y salida por las tres rutas (analitica, directa, FFT)."""
    dt = tau / pasos
    w1, w2 = factores[0] / tau, factores[1] / tau
    t_on = 60 * tau
    th, h = _h(tau, dt, 10 * tau)
    t, x = suma_sinusoides_encendida([1.0, 1.0], [w1, w2], t_on, dt, 80 * tau)

    yd = conv_directa(x, h, dt)[:len(t)]
    yf = conv_fft(x, h, dt)[:len(t)]
    ya = salida_analitica(1.0, w1, tau, t_on, t) + salida_analitica(1.0, w2, tau, t_on, t)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax1.plot(t * 1e3, x, color="gray",
             label=f"Entrada: $x(t)=\\sin({w1:.0f}\\,t)+\\sin({w2:.0f}\\,t)$,  $0 \\leq t < 60\\tau$")
    ax1.set_ylabel("x [V]")
    ax1.set_title("Filtro RC pasabajos con entrada de dos senoidales")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(t * 1e3, ya, label=r"Analítica: $y_1(t)+y_2(t)$, con $y_i=|H_i|[\sin(\omega_i t+\theta_i)-\sin\theta_i\,e^{-t/\tau}]$")
    ax2.plot(t * 1e3, yd, "--", label=r"Convolución directa: $y[n]=\Delta t\sum_k h[k]\,x[n-k]$")
    ax2.plot(t * 1e3, yf, ":", label=r"FFT: $y=\Delta t\,\mathcal{F}^{-1}\{X\,H\}$")
    ax2.set_xlabel("t [ms]")
    ax2.set_ylabel("V_C [V]")
    ax2.grid(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)

    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def figura_transitorio(tau=TAU, factores=(0.2, 1.0, 5.0), pasos=100, guardar=None):
    """Salida por FFT contra el regimen estacionario |H| sin(wt + theta).

    La franja gris (10 a 55 ms) es la ventana donde se mide amplitud y fase.
    """
    dt = tau / pasos
    t_on = 60 * tau
    th, h = _h(tau, dt, 10 * tau)

    fig, ejes = plt.subplots(len(factores), 1, sharex=True, figsize=(6.5, 5.5))
    for ax, f in zip(ejes, factores):
        w = f / tau
        t, x = sinusoide_encendida(1.0, w, t_on, dt, 80 * tau)
        y = conv_fft(x, h, dt)[:len(t)]
        H = H_analitica(w, tau)
        estacionario = np.where(t < t_on, abs(H) * np.sin(w * t + np.angle(H)), np.nan)
        ax.plot(t * 1e3, y, label="salida (FFT)")
        ax.plot(t * 1e3, estacionario, "--", label=r"estacionario $|H|\sin(\omega t+\theta)$")
        ax.axvspan(10, 55, color="gray", alpha=0.15)
        ax.set_ylabel("V_C [V]")
        ax.set_title(f"$\\omega\\tau$ = {f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)
    ejes[-1].set_xlabel("t [ms]")

    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig