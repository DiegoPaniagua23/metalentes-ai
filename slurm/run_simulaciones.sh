#!/bin/bash
# run_simulaciones.sh — Ejecutar simulaciones FDTD (Meep 2D) en job array y extraer features.
#
# Summary
# Cada task del array procesa una fila del CSV de geometrías: ejecuta el simulador,
# extrae el campo focal a JSON y elimina el .h5 intermedio si la extracción fue exitosa.
# La estrategia "Compute-Extract-Destroy" limita el uso de disco con ejecuciones paralelas.
#
# Usage
#   sbatch slurm/run_simulaciones.sh
#   sbatch --export=QUEUE_CSV=data/doe/simulations_queue_0.2.csv slurm/run_simulaciones.sh
#
# Parameters
#   QUEUE_CSV : path, optional
#     CSV de geometrías a simular. Default: data/doe/simulations_queue_0.1.csv.
#     Formato: header + filas  sim_id;widths;c2c_d  (separador ;).
#
# Inputs
#   data/doe/simulations_queue_0.1.csv   (o el CSV indicado por QUEUE_CSV)
#
# Outputs
#   data/raw/<SIM_ID>.h5                             buffer temporal (se elimina si JSON OK)
#   data/processed/simulations/<SIM_ID>_features.json
#   logs/%x-%A_%a.log
#
# Notes
#   --mem=0 solicita toda la memoria del nodo (Meep puede usar varios GB por simulación).
#   --array=1-500%5 limita a 5 tasks concurrentes para no saturar la partición GPU.
#   El .h5 crudo se conserva para depuración solo si extract_features.py falla.

#SBATCH --job-name=metalentes_sim
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrados_diego.paniagua/proyects/metalentes_ai
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --array=1-500%5

set -e  # Falla rápido si FDTD o extracción fallan; evita JSONs incompletos.

# Ruta fija de miniconda en Lab-SB; los nodos SLURM no cargan .bashrc del usuario.
export PATH="/home/est_posgrados_diego.paniagua/miniconda3/bin:$PATH"

# Sobrescribible por --export para seleccionar el lote (p.ej., simulations_queue_0.2.csv).
QUEUE_CSV=${QUEUE_CSV:-data/doe/simulations_queue_0.1.csv}

# El simulador y la extracción asumen directorios destino preexistentes.
mkdir -p logs data/raw data/processed/simulations

# SLURM_ARRAY_TASK_ID es 1-based; +1 salta el header del CSV.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$QUEUE_CSV")

# Columnas del CSV: sim_id;widths;c2c_d (separador ';').
SIM_ID=$(echo "$LINE" | awk -F';' '{print $1}')
WIDTHS=$(echo "$LINE" | awk -F';' '{print $2}')
C2C_D=$(echo "$LINE" | awk -F';' '{print $3}')

echo "==========================================================================="
echo "Job ID: $SLURM_JOB_ID | Array ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"
echo "Queue CSV: $QUEUE_CSV"
echo "Simulación: $SIM_ID"
echo "==========================================================================="

# Meep 2D usa threads y aprovecha --cpus-per-task=16 del job.
conda run -n consultoria_env python src/data_generation/Simulator_Metalen_Meep_2D.py \
    --sim_id "$SIM_ID" \
    --widths "$WIDTHS" \
    --c2c_d "$C2C_D" \
    --wavelength 0.630

echo "Simulación finalizada. Iniciando extracción de características..."

# extract_features.py lee el .h5 y escribe amplitud + fase focal re-muestreadas al JSON.
conda run -n consultoria_env python src/data_generation/extract_features.py \
    --file "data/raw/${SIM_ID}.h5" \
    --output_dir "data/processed/simulations"

# Eliminar el .h5 solo si el JSON existe; conservarlo permite re-extraer sin re-simular.
if [ -f "data/processed/simulations/${SIM_ID}_features.json" ]; then
    echo "JSON generado con éxito. Eliminando .h5 crudo (~230 MB) para liberar disco."
    rm "data/raw/${SIM_ID}.h5"
else
    echo "ADVERTENCIA: JSON no generado. Se conserva el .h5 para depuración."
fi

echo "Proceso completo para $SIM_ID."
