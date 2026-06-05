"""
Extraer metricas focales desde archivos HDF5 de campo generados por Meep.

Lee el campo complejo E_y del ultimo cuadro (estado estacionario), detecta el plano
focal por máximo de intensidad dentro de una ROI longitudinal, ajusta una parábola
a la trayectoria de picos para calcular R²_paper y exporta amplitud + fase al JSON.

Notas
-----
El vector 'phase' exportado corresponde al plano focal transversal (x_focal),
no a la fase de salida de las rendijas de la metalente.
"""

import h5py
import numpy as np
import json
import os
import argparse
from glob import glob


def extract_features(h5_filepath, output_dir):
    """
    Extrae R²_paper, amplitud y fase focal de un archivo .h5 de Meep.

    Ajusta un polinomio cuadrático a la trayectoria de picos de intensidad
    dentro de la ROI longitudinal y calcula R² del ajuste. La fase y la
    transmittance se extraen en la línea transversal del plano focal detectado.

    Parametros
    ----------
    h5_filepath : str
        Ruta al archivo .h5 generado por Simulator_Metalen_Meep_2D.py.
        Debe contener el dataset 'e_data' con shape (n_frames, nx, ny).
    output_dir : str
        Directorio donde se escribe <sim_id>_features.json.

    Devuelve
    --------
    None
        Escribe el JSON en output_dir e imprime R² y la ruta del archivo de salida.

    Notas
    -----
    ROI longitudinal: [0.20·nx, 0.95·nx). El 20% inicial excluye la zona de la
    metalente y la fuente; el 5% final evita artefactos del absorbedor PML.
    El epsilon 1e-9 en transmittance y en ss_tot previene división por cero
    cuando los picos no varían (geometría que no forma foco, ss_tot ≈ 0).
    'focal_quality' se conserva como etiqueta compacta de validez.
    Cualquier excepción se captura y se reporta por stdout sin propagar el error.
    """
    sim_id = os.path.basename(h5_filepath).replace('.h5', '')

    try:
        with h5py.File(h5_filepath, 'r') as f:
            # El ultimo cuadro representa el campo estacionario.
            e_data = f['e_data'][:]
            e_final = e_data[-1]
            nx, ny = e_final.shape

            intensity_2d = np.abs(e_final)**2

            # Excluir region de fuente/metalente y frontera absorbente.
            start_x = int(nx * 0.20)
            end_x = int(nx * 0.95)

            intensity_roi = intensity_2d[start_x:end_x, :]
            x_vals = np.arange(start_x, end_x)

            y_peaks = np.argmax(intensity_roi, axis=1)

            coeffs = np.polyfit(x_vals, y_peaks, 2)
            poly = np.poly1d(coeffs)
            y_pred = poly(x_vals)

            ss_res = np.sum((y_peaks - y_pred)**2)
            ss_tot = np.sum((y_peaks - np.mean(y_peaks))**2)
            # Guarda numerica para trayectorias de pico planas.
            r_squared = 1 - (ss_res / (ss_tot + 1e-9))

            # Plano focal: fila de ROI con maxima intensidad transversal.
            focal_x_rel = np.argmax(np.max(intensity_roi, axis=1))
            focal_x_abs = start_x + focal_x_rel

            e_line = e_final[focal_x_abs, :]

            intensity_line = np.abs(e_line)**2
            # Guarda numerica para geometrías sin enfoque.
            transmittance = intensity_line / (np.max(intensity_line) + 1e-9)
            phase_unwrapped = np.unwrap(np.angle(e_line))

            results = {
                "sim_id": sim_id,
                "r_squared": float(r_squared),
                "poly_coeffs": coeffs.tolist(),
                "focal_quality": "Valid" if r_squared >= 0.90 else "Invalid",
                "focal_plane_x": int(focal_x_abs),
                "transmittance": transmittance.tolist(),
                "phase": phase_unwrapped.tolist()
            }

            out_file = os.path.join(output_dir, f"{sim_id}_features.json")
            with open(out_file, 'w') as jf:
                json.dump(results, jf)

            print(f"[{sim_id}] R^2: {r_squared:.4f} | Calidad: {results['focal_quality']} -> {out_file}")

    except Exception as e:
        print(f"[{sim_id}] Error al procesar {h5_filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extracción de métricas focales")
    parser.add_argument("--file", type=str, default=None,
                        help="Ruta a un archivo .h5 específico")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="Directorio con archivos .h5 (ignorado si se usa --file)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directorio de salida para los .json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.file:
        if os.path.exists(args.file):
            extract_features(args.file, args.output_dir)
        else:
            print(f"Error: El archivo {args.file} no existe.")
        return

    h5_files = glob(os.path.join(args.input_dir, "*.h5"))
    if not h5_files:
        print(f"No se encontraron archivos .h5 en {args.input_dir}")
        return

    print(f"Encontrados {len(h5_files)} archivos para procesar.")
    for filepath in h5_files:
        extract_features(filepath, args.output_dir)

if __name__ == "__main__":
    main()
