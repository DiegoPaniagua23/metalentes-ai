# src/

Codigo vigente del pipeline. Cada subcarpeta representa una etapa y tiene su propio `README.md` con detalles locales.

## Mapa De Etapas

| Carpeta | Rol | Ejecuta local |
|---|---|---|
| `data_generation/` | DoE, simulacion FDTD y extraccion de features | Parcial |
| `data_processing/` | Construccion del dataset ML | Si |
| `surrogate/` | Entrenamiento y evaluacion del MLP | Si |
| `inverse_design/` | GA, calibracion, confirmacion y busqueda local | Parcial |

## Flujo Principal

```text
data_generation -> data_processing -> surrogate -> inverse_design -> slurm/FDTD
```

El README raiz contiene los comandos del pipeline completo. Este directorio documenta la responsabilidad de cada modulo.

## Restricciones Fisicas Usadas Por El Codigo

- `widths` en `[0.020, 0.300]` micras.
- `c2c_d` en `[0.150, 0.300]` micras.
- Pared minima de oro de `0.020` micras.
- Duty cycle entre `0.1` y `0.5`.

## Simulador

El simulador FDTD no se incluye en el repositorio y debe proporcionarse de manera personal para correr las etapas que lo requieren.
