"""
Ejecutar simulaciones FDTD secuenciales desde un CSV de geometrias.

Ejecuta la estrategia computar-extraer-eliminar: por cada geometría del CSV llama
run_simulation() (Meep 2D), luego extract_features() para escribir el JSON y elimina
el .h5 temporal. Si el JSON ya existe, la sim se omite, lo que permite
reanudar ejecuciones interrumpidas de forma idempotente.

Uso
---
python src/data_generation/generate_doe_0.1.py \
    --csv data/doe/simulations_queue_0.2.csv \
    --start 1 --end 500

python src/data_generation/generate_doe_0.1.py \
    --csv data/doe/simulations_queue_0.2.csv \
    --start 251 --end 500
"""

import csv
import os
import sys
import time
import argparse
import traceback

# Imports locales para ejecutar el script sin instalar el paquete.
sys.path.insert(0, os.path.dirname(__file__))

from Simulator_Metalen_Meep_2D import run_simulation
from extract_features import extract_features


def run_pipeline(csv_path, raw_dir, processed_dir, start, end, wavelength):
    """
    Ejecuta el pipeline FDTD (computar-extraer-eliminar) en el rango [start, end].

    Por cada geometría del CSV ejecuta run_simulation(), extract_features() y elimina .h5.
    Reanudación idempotente: si el JSON de features ya existe, la sim se omite.

    Parametros
    ----------
    csv_path : str
        Ruta al CSV con columnas sim_id;widths;c2c_d (separador ;).
    raw_dir : str
        Directorio para .h5 temporales (eliminados tras extracción).
    processed_dir : str
        Directorio de destino para los JSON de features.
    start : int
        Índice de inicio 1-based (primera fila de datos, sin contar el header).
    end : int
        Índice final 1-based, inclusivo.
    wavelength : float
        Longitud de onda en µm (0.630 para luz roja).

    Devuelve
    --------
    None
        Imprime resumen de completadas/fallidas. Los JSON se escriben en processed/.

    Notas
    -----
    En caso de excepción, el .h5 parcial se elimina para evitar lecturas corruptas
    en una re-ejecución posterior. El traceback completo se imprime para diagnóstico.
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)

    total = len(rows)
    rows_to_run = rows[start - 1:end]

    print(f"=== Pipeline Clúster Privado ===")
    print(f"CSV: {csv_path}")
    print(f"Total en CSV: {total} | Ejecutando: {start} a {end} ({len(rows_to_run)} sims)")
    print(f"Raw dir: {raw_dir} | Processed dir: {processed_dir}")
    print(f"=" * 40)

    completed = 0
    failed = 0

    for i, row in enumerate(rows_to_run):
        sim_id = row['sim_id']
        idx = start + i

        # Skip completed simulations.
        json_path = os.path.join(processed_dir, f"{sim_id}_features.json")
        if os.path.exists(json_path):
            print(f"[{idx}/{end}] {sim_id} ya procesada, saltando...")
            completed += 1
            continue

        widths = [float(x) for x in row['widths'].split(',')]
        c2c_d = [float(x) for x in row['c2c_d'].split(',')]

        h5_path = os.path.join(raw_dir, f"{sim_id}.h5")
        t0 = time.time()

        try:
            print(f"\n[{idx}/{end}] {sim_id} - Iniciando simulación...")
            run_simulation(sim_id, widths, c2c_d, wavelength)
            t_sim = time.time() - t0
            print(f"  Simulación completada en {t_sim:.1f}s")

            print(f"  Extrayendo features...")
            extract_features(h5_path, processed_dir)
            t_extract = time.time() - t0 - t_sim
            print(f"  Extracción completada en {t_extract:.1f}s")

            # Eliminar salida FDTD temporal despues de extraer features.
            if os.path.exists(h5_path):
                os.remove(h5_path)
                print(f"  {sim_id}.h5 eliminado ({t_sim + t_extract:.1f}s total)")

            completed += 1

        except Exception as e:
            failed += 1
            print(f"  ERROR en {sim_id}: {e}")
            traceback.print_exc()
            # Eliminar archivos parciales antes de reintentar.
            if os.path.exists(h5_path):
                os.remove(h5_path)
            continue

    print(f"\n{'=' * 40}")
    print(f"Completadas: {completed} | Fallidas: {failed} | Total: {completed + failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Runner de simulaciones para clúster privado (sin SLURM)")
    parser.add_argument("--csv", type=str,
                        default="data/doe/simulations_queue_0.2.csv",
                        help="Ruta al CSV de configuraciones")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Directorio para archivos .h5 temporales")
    parser.add_argument("--processed_dir", type=str, default="data/processed/simulations",
                        help="Directorio para archivos .json de features")
    parser.add_argument("--start", type=int, default=1,
                        help="Índice de inicio (1-indexed)")
    parser.add_argument("--end", type=int, default=500,
                        help="Índice final (inclusive)")
    parser.add_argument("--wavelength", type=float, default=0.630,
                        help="Longitud de onda en μm")
    args = parser.parse_args()

    run_pipeline(args.csv, args.raw_dir, args.processed_dir,
                 args.start, args.end, args.wavelength)


if __name__ == "__main__":
    main()
