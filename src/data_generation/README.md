# src/data_generation/

Scripts relacionados con generacion de geometrias y extraccion de features FDTD.

## Archivos

| Archivo | Rol |
|---|---|
| `generate_doe_0.1.py` | Genera el lote DoE `queue_0.1` |
| `generate_doe_0.2.py` | Genera el lote DoE `queue_0.2` |
| `extract_features.py` | Extrae `R2_real`, intensidad y fase desde una simulacion FDTD |

## Simulador

El simulador FDTD no se incluye en este repositorio y debe proporcionarse de manera personal. Sin ese archivo, esta etapa no puede ejecutar FDTD, pero el resto del pipeline puede revisarse y ejecutarse con datos ya generados.

## Salidas Esperadas

- CSVs de geometria en `data/doe/`.
- JSONs de features en `data/processed/` o `data/confirmed/`.

## Nota Sobre La Fase

El vector `phase` extraido corresponde a la fase del campo en el plano focal. No representa la fase a la salida de cada rendija.
