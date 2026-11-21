# Opción D: filtro RC pasabajos en tiempo y frecuencia

Universidad del Quindío, Programa de Física.
Trabajo final de Matemáticas Especiales y Programación.

**Integrantes:** [Ximena Criollo Grisales], [Alejandro Restrepo Aguirre]

## Qué hace el proyecto

Calcula la respuesta de un filtro RC pasabajos a una senoidal encendida y a la suma de
dos senoidales, de tres maneras: convolución directa, convolución por FFT y solución
analítica. Compara los resultados y mide la amplitud y la fase de la salida en régimen
estacionario contra H(w) = 1/(1 + i w tau).

## Parámetros usados

- tau = RC = 1 ms (R = 1 kOhm, C = 1 uF)
- dt = tau/100 (figura compuesta y tabla de métodos) y tau/200 (demás tablas)
- h truncada en 10 tau, con h[0]/2 en el salto de t = 0
- tres senoidales simples con w*tau = 0.2, 1 y 5
- señal compuesta con w*tau = 0.2 y 5, encendida entre 0 y 60 ms

## Cómo ejecutarlo

1. Crear el entorno e instalar dependencias:

       python -m venv .venv
       .venv\Scripts\Activate.ps1
       pip install -r requirements.txt

2. Abrir notebook_resultados.ipynb y ejecutar todas las celdas en orden.

## Estructura

- src/sistema.py: h(t) y H(w)
- src/entradas.py: senoidal encendida y suma de senoidales
- src/convolucion.py: convolución directa, convolución por FFT y eje de salida
- src/validacion.py: solución analítica y estimación de amplitud y fase
- src/resultados.py: tablas de resultados con pandas
- notebook_resultados.ipynb: modelo, figuras, tablas y conclusiones
- figura_H.png y figura_compuesta.png: figuras generadas por el notebook
- prueba1.py a prueba12.py: pruebas exploratorias hechas durante el desarrollo

## Dependencias

Ver requirements.txt (numpy, scipy, matplotlib, pandas, jupyter, ipykernel).