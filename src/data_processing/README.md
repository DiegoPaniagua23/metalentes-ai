# src/data_processing/

Construye el dataset ML a partir de JSONs FDTD y geometrias.

## Orden De Uso

```bash
python src/data_processing/resample_focal_fields.py
python src/data_processing/extend_dataset_iter3.py
python src/data_processing/extend_dataset_iter4.py
```

## Archivos

| Archivo | Rol |
|---|---|
| `resample_focal_fields.py` | Consolida geometrias, amplitud, fase y R2 a longitud fija |
| `extend_dataset_iter3.py` | Agrega confirmados iter 3 y pesos de muestra |
| `extend_dataset_iter4.py` | Agrega confirmados iter 4 y actualiza dataset v5 |
| `build_mvp_dataset.py` | Constructor alternativo de splits del dataset |
| `propagation.py` | Propagacion angular spectrum para validaciones auxiliares |

## Salidas

Artefactos en `data/dataset/`:

- `fields.npz`
- `splits.npz`
- `dataset_train.npz`
- `dataset_val.npz`
- `dataset_test.npz`
- `metadata.json`

Estos archivos son regenerables y no se versionan.

## Convenciones

- Longitud fija del campo: `1024`.
- Entrada `X`: 101 anchos + 1 `c2c_d`.
- La fase se centra por simulacion antes de entrenar.
