# data/

Datos usados por el pipeline. Esta carpeta separa datos versionados ligeros de artefactos pesados/regenerables ignorados por Git.

## Estructura

```text
data/
|-- doe/           # CSVs con geometrias de entrada
|-- processed/     # JSONs masivos de features FDTD, ignorados
|-- confirmed/     # JSONs confirmados por FDTD, versionados
|-- dataset/       # Dataset ML regenerable; .npz ignorados
`-- _archive/      # Material historico local, ignorado
```

## Que Se Versiona

- `data/doe/*.csv`: geometrias base del diseno de experimentos.
- `data/confirmed/**/*.json`: candidatos evaluados por FDTD real. Son ligeros y documentan `R2_real`.
- `data/dataset/provenance.csv`: trazabilidad de simulaciones usadas para construir el dataset.

## Que No Se Versiona

- `data/processed/**/*.json`: features masivos regenerables.
- `*.h5` y `*.pickle`: salidas crudas/intermedias FDTD.
- `data/dataset/*.npz` y `metadata.json`: dataset consolidado regenerable.
- `data/_archive/`: material historico local.

## Campos Importantes En JSONs Confirmados

| Campo | Significado |
|---|---|
| `cand_id` o `sim_id` | Identificador del candidato o simulacion |
| `r_squared` | `R2_real` calculado con FDTD |
| `focal_quality` | `Valid` si `r_squared >= 0.90` |
| `widths` | 101 anchos de rendija en micras |
| `c2c_d` | Separacion centro a centro en micras |
| `transmittance` | Perfil de intensidad normalizado |
| `phase` | Fase desenvuelta en el plano focal |

## Regenerar Dataset ML

Desde la raiz del repo:

```bash
python src/data_processing/resample_focal_fields.py
python src/data_processing/extend_dataset_iter3.py
python src/data_processing/extend_dataset_iter4.py
```

Esto reconstruye los artefactos de `data/dataset/` usados por el surrogate.

## Resultado Final En Datos

Los candidatos robustos principales estan en `data/confirmed/iterhc627/` y resumidos en `results/final/`.
