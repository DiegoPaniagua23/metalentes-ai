# slurm/

Scripts para ejecutar etapas FDTD en cluster.

## Archivos

| Archivo | Rol |
|---|---|
| `run_simulaciones.sh` | Ejecuta simulaciones FDTD de un CSV de geometrias |
| `slurm_confirm_mvp.sh` | Confirma candidatos generados por el GA o hill-climbing |

## Confirmar Candidatos

```bash
sbatch --export=ITER=N slurm/slurm_confirm_mvp.sh
```

Variables utiles:

| Variable | Uso |
|---|---|
| `ITER` | Carpeta de salida `data/confirmed/iter<ITER>/` |
| `CSV_PATH` | CSV de candidatos alternativo |
| `END` | Limite superior de candidatos a procesar |

## Salida

JSONs de features en `data/confirmed/iterN/`.

## Requisito

El simulador FDTD no se incluye en este repositorio y debe proporcionarse de manera personal en el entorno del cluster.
