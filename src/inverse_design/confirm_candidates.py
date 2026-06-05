"""
Confirm inverse-design candidates with FDTD.

Lee un CSV con geometrías candidatas, simula cada una con Meep, extrae features
(R²_real) y guarda los JSON en --output_dir. El layout estandarizado escribe
directamente a data/confirmed/iter<N>/.

Formato esperado del CSV
------------------------
cand_id;widths;c2c_d;r2_pred;fwhm_pred;concentration_pred;fitness[;seed_origin]

Uso (en cluster)
-----------------
python src/inverse_design/confirm_candidates.py \
    --candidates_csv results/inverse_design/top_candidates.csv \
    --output_dir     data/confirmed/iter5 \
    --start 1 --end 10
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_generation'))
from Simulator_Metalen_Meep_2D import run_simulation
from extract_features import extract_features


def main():
    """
    Pipeline FDTD para confirmar candidatos: lee CSV, simula con Meep, extrae R²_real.

    Por cada fila del CSV ejecuta run_simulation() (Meep 2D) seguido de
    extract_features() para obtener R²_paper real. Es idempotente: si el JSON
    de un cand_id ya existe en --output_dir se omite la simulación. El .h5 temporal
    se elimina post-extracción para mantener el disco acotado en el cluster
    (~230 MB por sim).

    Devuelve
    --------
    None
        Imprime resumen tabular (cand_id, R²_pred, R²_real, Δ, estado) al stdout.
        Los JSON enriquecidos con r2_pred_mvp, cand_id, widths y c2c_d se escriben
        en --output_dir.
    """
    parser = argparse.ArgumentParser(description="Confirma candidatos via FDTD")
    parser.add_argument("--candidates_csv", default="results/inverse_design/top_candidates.csv")
    parser.add_argument("--output_dir",     default="data/confirmed/iter_manual")
    parser.add_argument("--raw_dir",        default="data/raw")
    parser.add_argument("--start",          type=int, default=1)
    parser.add_argument("--end",            type=int, default=10)
    parser.add_argument("--wavelength",     type=float, default=0.630)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    # Leer filas de candidatos en formato de salida del GA.
    with open(args.candidates_csv) as f:
        candidates = list(csv.DictReader(f, delimiter=';'))

    candidates_to_run = candidates[args.start - 1:args.end]

    print("=" * 70)
    print(" CONFIRMACION DE CANDIDATOS CON FDTD")
    print("=" * 70)
    print(f"  CSV:        {args.candidates_csv}")
    print(f"  Candidatos: {args.start} a {args.end} ({len(candidates_to_run)} sims)")
    print(f"  Output:     {args.output_dir}/")
    print(f"  Criterio éxito propuesta: R²_real ≥ 0.90")
    print()

    completed, failed, validated = 0, 0, 0
    results_summary = []

    for row in candidates_to_run:
        cand_id    = row['cand_id']
        r2_pred    = float(row['r2_pred'])
        c2c_d_val  = float(row['c2c_d'])
        widths     = [float(x) for x in row['widths'].split(',')]
        c2c_d_list = [c2c_d_val] * (len(widths) - 1)
        sim_id = cand_id

        # Skip completed confirmations.
        json_path = os.path.join(args.output_dir, f"{sim_id}_features.json")
        if os.path.exists(json_path):
            print(f"\n[{cand_id}] ya confirmado, saltando...")
            with open(json_path) as f:
                d = json.load(f)
            if d.get('r_squared', 0) >= 0.90:
                validated += 1
            results_summary.append({**d, 'cand_id': cand_id, 'r2_pred': r2_pred})
            completed += 1
            continue

        h5_path = os.path.join(args.raw_dir, f"{sim_id}.h5")
        t0 = time.time()

        try:
            print(f"\n[{cand_id}]  R²_pred={r2_pred:.4f} | c2c_d={c2c_d_val:.4f} | "
                  f"w_mean={sum(widths)/len(widths):.4f} | w_min={min(widths):.4f} w_max={max(widths):.4f}")
            print(f"  Simulando con FDTD...")
            run_simulation(sim_id, widths, c2c_d_list, args.wavelength)
            t_sim = time.time() - t0
            print(f"  Simulacion: {t_sim:.1f}s")

            print(f"  Extrayendo features...")
            extract_features(h5_path, args.output_dir)
            t_total = time.time() - t0
            print(f"  Extraccion: {t_total - t_sim:.1f}s | Total: {t_total:.1f}s")

            # Persistir metadatos del candidato junto con las features FDTD.
            with open(json_path) as f:
                result = json.load(f)
            result['r2_pred_mvp'] = r2_pred
            result['cand_id'] = cand_id
            result['widths'] = widths
            result['c2c_d'] = c2c_d_val
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)

            r2_real = result['r_squared']
            is_valid = r2_real >= 0.90
            marker = "VALID" if is_valid else "Invalid"
            print(f"  R²_real = {r2_real:.4f}  vs  R²_pred = {r2_pred:.4f}  ->  {marker}")
            if is_valid:
                validated += 1
            results_summary.append({**result, 'cand_id': cand_id, 'r2_pred': r2_pred})

            if os.path.exists(h5_path):
                os.remove(h5_path)

            completed += 1

        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
            traceback.print_exc()
            if os.path.exists(h5_path):
                os.remove(h5_path)

    print(f"\n{'=' * 70}")
    print(f" RESUMEN")
    print(f"{'=' * 70}")
    print(f"  Completados: {completed}/{len(candidates_to_run)} | Fallidos: {failed}")
    print(f"  Con R²_real ≥ 0.90: {validated}/{completed}")
    print()
    if results_summary:
        print(f"  {'cand_id':<12}{'R²_pred':<10}{'R²_real':<10}{'Δ':<10}{'estado':<12}")
        print("  " + "-" * 54)
        for r in results_summary:
            r2_real = r.get('r_squared', 0)
            r2_pred = r.get('r2_pred', 0)
            delta = r2_pred - r2_real
            estado = "VALID" if r2_real >= 0.90 else "Invalid"
            print(f"  {r['cand_id']:<12}{r2_pred:<10.4f}{r2_real:<10.4f}{delta:+<10.4f}{estado:<12}")
    print()


if __name__ == "__main__":
    main()
