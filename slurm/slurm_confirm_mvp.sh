#!/bin/bash
# slurm_confirm_mvp.sh — Confirmar candidatos del GA o del hill-climbing por FDTD en Lab-SB.
#
# Summary
# Ejecuta confirm_candidates.py sobre un rango de filas de un CSV de candidatos.
# Cada fila define una geometría (widths + c2c_d) que Meep simula en 2D; luego
# extract_features.py genera JSONs con R²_real (criterio de éxito final).
#
# Usage
#   # GA top-10 (1 job):
#   sbatch --export=ITER=5 slurm/slurm_confirm_mvp.sh
#
#   # Hill-climbing por cuenca (1 job por cuenca, lanzar en paralelo):
#   sbatch --export=ITER=hc042,CSV_PATH=results/inverse_design/top_candidates_hc042.csv slurm/slurm_confirm_mvp.sh
#   sbatch --export=ITER=hc721,CSV_PATH=results/inverse_design/top_candidates_hc721.csv slurm/slurm_confirm_mvp.sh
#   sbatch --export=ITER=hc627,CSV_PATH=results/inverse_design/top_candidates_hc627.csv slurm/slurm_confirm_mvp.sh
#
# Parameters (vía --export=…)
#   ITER     : REQUERIDO. Tag de iteración. Output → data/confirmed/iter${ITER}/ (o iter5, iterhc042, etc.).
#   CSV_PATH : opcional. CSV de candidatos. Default: results/inverse_design/top_candidates.csv.
#   START    : opcional. Default: 1.
#   END      : opcional. Default: 10 (GA) o el número de filas del CSV menos 1.
#
# Outputs
#   data/confirmed/iter${ITER}/   JSONs con R²_real (recuperar al local con rsync)
#   data/raw/                     .h5 crudos (eliminados post-extracción)
#   logs/%x-%j.log

#SBATCH --job-name=mvp_confirm
#SBATCH --partition=C0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/est_posgrados_diego.paniagua/proyects/metalentes_ai
#SBATCH --output=logs/%x-%j.log

set -e

export PATH="/home/est_posgrados_diego.paniagua/miniconda3/bin:$PATH"

if [[ -z "${ITER:-}" ]]; then
    echo "ERROR: variable ITER no especificada. Usa: sbatch --export=ITER=N slurm/slurm_confirm_mvp.sh"
    exit 1
fi

CSV_PATH=${CSV_PATH:-results/inverse_design/top_candidates.csv}
START=${START:-1}
END=${END:-10}
OUTPUT_DIR="data/confirmed/iter${ITER}"

mkdir -p logs results data/raw "${OUTPUT_DIR}"

echo "==========================================================================="
echo " CONFIRMACION MVP — Job $SLURM_JOB_ID  |  Host $(hostname)"
echo "==========================================================================="
echo "  Iteración:         ${ITER}"
echo "  CSV de geometrías: ${CSV_PATH}"
echo "  Rango candidatos:  $START a $END"
echo "  Output:            ${OUTPUT_DIR}/"
echo "==========================================================================="

conda run -n consultoria_env python src/inverse_design/confirm_candidates.py \
    --candidates_csv "${CSV_PATH}" \
    --output_dir     "${OUTPUT_DIR}" \
    --raw_dir        data/raw \
    --start          $START \
    --end            $END \
    --wavelength     0.630

echo "==========================================================================="
echo " Confirmación finalizada. Resultados en ${OUTPUT_DIR}/"
echo " Próximo paso: rsync ${OUTPUT_DIR}/ → local data/confirmed/iter${ITER}/"
echo "==========================================================================="