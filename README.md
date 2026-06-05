# Diseño de Metalentes Usando Inteligencia Artificial

Repositorio de transferencia del pipeline usado para disenar metalentes plasmonicas de 101 nano-rendijas mediante un surrogate de ML y confirmacion FDTD.

El criterio fisico del proyecto es alcanzar `R2 >= 0.90` en el ajuste parabolico del perfil de intensidad, confirmado con simulacion FDTD real.

## Resultado Principal

| Candidato | R2_real FDTD | Uso recomendado |
|---|---:|---|
| `hc627_036` | 0.9256 | Geometria recomendada |
| `hc627_008` | 0.9134 | Alternativa robusta |
| `hc627_043` | 0.9014 | Alternativa robusta |

Los resultados finales estan en `results/final/`.

## Que Incluye Este Repositorio

Incluye:

- Codigo vigente del pipeline en `src/`.
- Scripts SLURM vigentes en `slurm/`.
- Archivos fuente del reporte en `report/`.
- Resultados finales consolidados en `results/final/`.
- JSONs FDTD confirmados en `data/confirmed/`.
- Definicion del entorno en `consultoria_env.yml`.

No incluye:

- Simulaciones FDTD crudas o procesadas masivas.
- Datasets `.npz` regenerables.
- Modelos entrenados.
- Logs.
- Notas de trabajo locales.
- El simulador FDTD. Debe proporcionarse de manera personal para correr las etapas de simulacion y confirmacion.

## Estructura

```text
metalentes_ai/
|-- data/                 # DoE, JSONs confirmados y trazabilidad de datos
|-- src/                  # Codigo del pipeline
|-- slurm/                # Ejecucion FDTD en cluster
|-- results/              # Resultados finales y artefactos regenerables
|-- report/               # Reporte LaTeX y figuras
|-- consultoria_env.yml   # Entorno reproducible
`-- README.md             # Punto de entrada
```

Cada carpeta principal tiene su propio `README.md` con alcance local.

## Instalacion

```bash
micromamba env create -f consultoria_env.yml
micromamba activate consultoria_env
python -c "import numpy, scipy, sklearn, torch; print('OK')"
```

Meep/FDTD se ejecuta solo en el cluster donde este disponible el simulador.

## Pipeline Reproducible

El flujo completo es:

```text
DoE CSV -> FDTD -> JSON features -> dataset ML -> surrogate -> GA -> FDTD confirmacion -> resultados finales
```

### 1. Preparar dataset ML

Requiere los JSONs de features en `data/processed/` y `data/confirmed/`.

```bash
python src/data_processing/resample_focal_fields.py
python src/data_processing/extend_dataset_iter3.py
python src/data_processing/extend_dataset_iter4.py
```

Salida esperada: `data/dataset/*.npz` y `data/dataset/metadata.json`.

### 2. Entrenar y evaluar surrogate

```bash
python src/surrogate/train_field_surrogate.py \
  --epochs 200 --batch_size 64 --hidden_dims 512,512,512 \
  --dropout 0.1 --lr 1e-3 \
  --weight_amp 1.0 --weight_phase 1.0 --weight_r2 10.0

python src/surrogate/evaluate_field_surrogate.py
```

Salida esperada: `results/surrogate/model.pt`, `training_log.json`, `eval_metrics.json`.

### 3. Optimizar geometria

```bash
python src/inverse_design/design_optimizer.py \
  --multi_seed_from_jsons data/confirmed/iter4 \
  --seed_min_r2 0.85 \
  --fitness_mode combo \
  --combo_calibration results/inverse_design/combo_calibration.json \
  --max_deviation 0.010 \
  --diverse_top_k_by_cluster \
  --pop_size 250 --n_gen 80 --top_k 10
```

Salida esperada: `results/inverse_design/top_candidates.csv`.

### 4. Confirmar por FDTD

En cluster:

```bash
sbatch --export=ITER=N slurm/slurm_confirm_mvp.sh
```

Salida esperada: `data/confirmed/iterN/*_features.json` con `r_squared` real.

## Lectura Recomendada

- `data/README.md`: que datos existen, cuales se regeneran y como se interpretan.
- `src/README.md`: mapa de scripts por etapa.
- `slurm/README.md`: ejecucion en cluster.
- `results/README.md`: significado de resultados versionados.
- `report/README.md`: compilacion del reporte.

## Convenciones

- `R2_paper`: metrica fisica del ajuste parabolico usada como criterio de exito.
- `R2_pred`: prediccion del surrogate.
- `R2_real`: valor confirmado por FDTD.
- `MSE_field`: error del surrogate al predecir campo focal.

## Estado

La etapa experimental esta cerrada. El repo conserva el pipeline que produjo el resultado principal y los artefactos necesarios para reproducir el flujo computacional cuando se cuente con los datos y el simulador requeridos.
