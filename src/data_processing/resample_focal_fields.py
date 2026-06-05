"""
Remuestrear features JSON de campo focal a un dataset ML de longitud fija.

Lee los JSONs históricos de data/processed/ y data/confirmed/, extrae los vectores
(transmittance, phase) en el plano focal y los re-muestrea a longitud fija (default 1024).
Cruza con queue CSVs para obtener la geometría correspondiente (widths, c2c_d).

Salidas
-------
data/dataset/fields.npz
    X:           (N, 102)   widths (101) + c2c_d (1) por sim
    amplitude:   (N, L)     |E|(y) normalizado, re-muestreado a L=1024
    phase:       (N, L)     Phi(y) desenvuelta y centrada, re-muestreada a L=1024
    r_squared:   (N,)       R2_paper (referencia, no es target ML)
    sim_ids:     (N,)       identificadores

Notas
-----
- amplitude = sqrt(transmittance), normalizada al pico de cada sim.
- phase se centra restando la mediana por sim (offset global no afecta foco).
- External Simulation_XXX records are excluded from training.
"""

import argparse
import csv
import json
import os
from glob import glob
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


def parse_geometry_field(raw):
    """
    Convertir representaciones varias a lista de floats.

    Parametros
    ----------
    raw : Any
        Representación del campo de geometría.

    Devuelve
    --------
    list of float
        Lista parseada; vacía si no se pudo interpretar.

    Notas
    -----
    Acepta:
    - lista Python: [0.05, 0.07, ...]
    - lista en str con corchetes: "[0.05, 0.07, ...]"
    - lista en str sin corchetes: "0.05,0.07,..."
    - escalar str: "0.234"
    - escalar numérico: 0.234
    """
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith('['):
        raw = raw.strip('[]')
    # Queue CSVs store vector fields as comma-separated strings.
    parts = [p.strip() for p in raw.split(',')]
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def load_queue_csv(path: str) -> dict:
    """
    Leer queue CSV con formato sim_id;widths;c2c_d (separador ; o ,).

    Parametros
    ----------
    path : str
        Ruta al CSV.

    Devuelve
    --------
    dict
        Mapeo sim_id -> {'widths': list[float], 'c2c_d': float}.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        first = f.readline()
        sep = ';' if ';' in first else ','
        f.seek(0)
        reader = csv.DictReader(f, delimiter=sep)
        for row in reader:
            sim_id = row.get('sim_id') or row.get('id')
            if not sim_id:
                continue
            try:
                widths = parse_geometry_field(row.get('widths', '[]'))
                c2c_d_list = parse_geometry_field(row.get('c2c_d', '[]'))
                if not widths or not c2c_d_list:
                    continue
                # c2c_d may be scalar or a constant-length list.
                c2c_d_scalar = float(np.mean(c2c_d_list))
                out[sim_id] = {'widths': widths, 'c2c_d': c2c_d_scalar}
            except Exception:
                continue
    return out


def load_json_features(json_path: str) -> Optional[dict]:
    """
    Cargar features de un JSON.

    Parametros
    ----------
    json_path : str
        Ruta al JSON de features.

    Devuelve
    --------
    dict or None
        Diccionario con phase, transmittance y r_squared si es válido.

    Notas
    -----
    Si el JSON tiene widths/c2c_d embebidos, se devuelven para evitar buscar en queue.
    """
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        return None

    phase = d.get('phase')
    trans = d.get('transmittance')
    r2 = d.get('r_squared')
    if phase is None or trans is None or r2 is None:
        return None
    if len(phase) != len(trans):
        return None
    return {
        'sim_id': d.get('sim_id') or os.path.basename(json_path).replace('_features.json', '').replace('.json', ''),
        'phase': np.asarray(phase, dtype=np.float64),
        'transmittance': np.asarray(trans, dtype=np.float64),
        'r_squared': float(r2),
        'embedded_widths': d.get('widths'),
        'embedded_c2c_d': d.get('c2c_d'),
    }


def resample_vector(vec: np.ndarray, target_length: int, kind: str = 'cubic') -> np.ndarray:
    """
    Re-muestrear un vector 1D a longitud fija vía interpolación.

    Parametros
    ----------
    vec : ndarray of float, shape (n,)
        Vector de entrada.
    target_length : int
        Longitud objetivo.
    kind : str, optional
        Tipo de interpolación ("cubic" o "linear" en fallback).

    Devuelve
    --------
    ndarray of float, shape (target_length,)
        Vector re-muestreado.

    Notas
    -----
    Mapea el eje original [0, n-1] a [0, target_length-1] uniformemente.
    """
    n = len(vec)
    if n == target_length:
        return vec.astype(np.float64)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    # 'cubic' requiere n >= 4
    if n < 4:
        kind = 'linear'
    f = interp1d(x_old, vec, kind=kind, assume_sorted=True)
    return f(x_new)


def build_sample(features: dict, geometry: dict, target_length: int, interp_kind: str) -> Optional[dict]:
    """
    Construir una muestra (X, |E|, Φ, R², ny_original) desde features y geometría.

    Parametros
    ----------
    features : dict
        Diccionario con phase, transmittance y r_squared.
    geometry : dict
        Diccionario con widths (len 101) y c2c_d (float).
    target_length : int
        Longitud objetivo para el re-muestreo.
    interp_kind : str
        Tipo de interpolación.

    Devuelve
    --------
    dict or None
        Muestra consolidada o None si no cumple restricciones (n_slits != 101).
    """
    widths = geometry['widths']
    c2c_d = geometry['c2c_d']

    if len(widths) != 101:
        return None

    trans = features['transmittance']
    phase = features['phase']
    ny_original = len(phase)

    # Remuestrear ambas componentes de campo a longitud fija.
    trans_rs = resample_vector(trans, target_length, kind=interp_kind)
    phase_rs = resample_vector(phase, target_length, kind=interp_kind)

    # Cubic interpolation can create small overshoots.
    trans_rs = np.clip(trans_rs, 0.0, 1.0)
    amplitude = np.sqrt(trans_rs)

    # Eliminar offset global de fase por simulacion.
    phase_centered = phase_rs - np.median(phase_rs)

    # Geometry vector: 101 widths + scalar c2c_d.
    X = np.concatenate([np.asarray(widths, dtype=np.float64), [c2c_d]])

    return {
        'sim_id': features['sim_id'],
        'X': X,
        'amplitude': amplitude,
        'phase': phase_centered,
        'r_squared': features['r_squared'],
        'ny_original': ny_original,
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-procesa JSONs a dataset de campo focal.")
    parser.add_argument('--processed_dir', default='data/processed/simulations',
                        help='JSONs de sim_XXX (Lab-SB + Privado). NO incluye simulations_extra (consultora, benchmark OOD).')
    parser.add_argument('--confirmed_dir', default='data/confirmed/al_ga',
                        help='JSONs confirmados previos (geometría embebida).')
    parser.add_argument('--queue_csv_lab', default='data/doe/simulations_queue_0.1.csv', help='Geometría sim_001-500')
    parser.add_argument('--queue_csv_priv', default='data/doe/simulations_queue_0.2.csv', help='Geometría sim_501-1000')
    parser.add_argument('--target_length', type=int, default=1024, help='Longitud fija para resampling')
    parser.add_argument('--interp_kind', default='cubic', choices=['linear', 'cubic', 'quadratic'])
    parser.add_argument('--output_dir', default='data/dataset', help='Salida (fields.npz)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(" RE-MUESTREO DE CAMPOS FOCALES")
    print("=" * 80)
    print(f"  target_length = {args.target_length}")
    print(f"  interpolación = {args.interp_kind}")
    print()

    # Cargar colas de geometria.
    print("[1/3] Cargando queue CSVs...")
    queue_lab = load_queue_csv(args.queue_csv_lab)
    queue_priv = load_queue_csv(args.queue_csv_priv)
    queue_combined = {**queue_lab, **queue_priv}
    print(f"  queue_lab:  {len(queue_lab)} entradas")
    print(f"  queue_priv: {len(queue_priv)} entradas")
    print(f"  combinado:  {len(queue_combined)} entradas")
    print()

    # Procesar features JSON historicas y confirmadas.
    print("[2/3] Procesando JSONs...")
    json_files = sorted(glob(os.path.join(args.processed_dir, '*.json'))) + \
                 sorted(glob(os.path.join(args.confirmed_dir, '*.json')))

    samples = []
    n_skipped = {'no_features': 0, 'no_geometry': 0, 'wrong_n_slits': 0, 'parse_error': 0}

    for jp in json_files:
        feats = load_json_features(jp)
        if feats is None:
            n_skipped['no_features'] += 1
            continue

        # Resolver geometria desde campos embebidos o CSVs de cola.
        if feats['embedded_widths'] is not None and feats['embedded_c2c_d'] is not None:
            c2c_raw = feats['embedded_c2c_d']
            c2c_scalar = float(np.mean(c2c_raw)) if isinstance(c2c_raw, list) else float(c2c_raw)
            geometry = {'widths': list(feats['embedded_widths']), 'c2c_d': c2c_scalar}
        elif feats['sim_id'] in queue_combined:
            geometry = queue_combined[feats['sim_id']]
        else:
            n_skipped['no_geometry'] += 1
            continue

        sample = build_sample(feats, geometry, args.target_length, args.interp_kind)
        if sample is None:
            n_skipped['wrong_n_slits'] += 1
            continue

        samples.append(sample)

    print(f"  JSONs procesados:  {len(json_files)}")
    print(f"  Muestras válidas:  {len(samples)}")
    print(f"  Descartadas:")
    for k, v in n_skipped.items():
        print(f"    - {k}: {v}")
    print()

    # Consolidate arrays.
    print("[3/3] Consolidando arrays...")
    N = len(samples)
    L = args.target_length
    sim_ids = np.array([s['sim_id'] for s in samples])
    X = np.stack([s['X'] for s in samples])
    amplitude = np.stack([s['amplitude'] for s in samples])
    phase = np.stack([s['phase'] for s in samples])
    r_squared = np.array([s['r_squared'] for s in samples])
    ny_original = np.array([s['ny_original'] for s in samples])

    print(f"  X shape:         {X.shape}    dtype={X.dtype}")
    print(f"  amplitude shape: {amplitude.shape}    dtype={amplitude.dtype}")
    print(f"  phase shape:     {phase.shape}    dtype={phase.dtype}")
    print(f"  r_squared shape: {r_squared.shape}")
    print()

    # Basic numerical checks.
    print(" Sanity checks:")
    print(f"  X.min()={X.min():.4f}, X.max()={X.max():.4f}")
    print(f"  amplitude: range [{amplitude.min():.4f}, {amplitude.max():.4f}], mean {amplitude.mean():.4f}")
    print(f"  phase:     range [{phase.min():.4f}, {phase.max():.4f}], mean {phase.mean():.4f}")
    print(f"  r_squared: range [{r_squared.min():.4f}, {r_squared.max():.4f}], mean {r_squared.mean():.4f}")
    print(f"  R²≥0.90:  {(r_squared >= 0.90).sum()}/{N}")
    print(f"  NaN check: amplitude={np.isnan(amplitude).sum()}, phase={np.isnan(phase).sum()}, X={np.isnan(X).sum()}")
    print()

    out_path = os.path.join(args.output_dir, 'fields.npz')
    np.savez_compressed(out_path,
                        sim_ids=sim_ids, X=X, amplitude=amplitude, phase=phase, r_squared=r_squared,
                        ny_original=ny_original, target_length=L)
    print(f"Guardado: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print()


if __name__ == '__main__':
    main()
