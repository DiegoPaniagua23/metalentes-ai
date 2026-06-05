# src/inverse_design/

Optimizacion inversa y confirmacion de candidatos.

## Archivos

| Archivo | Rol |
|---|---|
| `design_optimizer.py` | GA manifold-restricted y generacion de candidatos |
| `calibrate_iter3_fitness.py` | Calibra fitness combo usando iter 3 |
| `calibrate_iter4_fitness.py` | Calibracion exploratoria posterior |
| `confirm_candidates.py` | Confirma candidatos con FDTD |
| `random_search_local.py` | Hill-climbing FDTD directo alrededor de cuencas |

## Uso Local Principal

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

## Salidas

- `results/inverse_design/top_candidates.csv`
- `results/inverse_design/ga_convergence.json`

Estos outputs son regenerables y no se versionan.

## Confirmacion FDTD

`confirm_candidates.py` requiere el simulador FDTD. Normalmente se ejecuta desde `slurm/slurm_confirm_mvp.sh`.
