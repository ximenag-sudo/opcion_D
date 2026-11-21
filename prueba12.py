from src.resultados import (tabla_metodos, tabla_amplitud_fase,
                            tabla_truncamiento, tabla_paso)

for nombre, f in [("metodos", tabla_metodos),
                  ("amplitud y fase", tabla_amplitud_fase),
                  ("truncamiento", tabla_truncamiento),
                  ("paso dt", tabla_paso)]:
    print("\n" + nombre)
    print(f().to_string(index=False))