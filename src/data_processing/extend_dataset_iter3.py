"""
Extend the base dataset with iteration 3 confirmations.

Toma el fields.npz base (1090 sims, en data/dataset/) y agrega los
10 confirmados de data/confirmed/iter3/ (re-muestreados a L=1024 con la misma
pipeline que resample_focal_fields.py).

Asigna sample_weight por simulación:
- iter 3 con R²_real ≥ 0.85 (mvp_c01, c02, c03, c08, c10):  ×30
- iter 3 con R²_real <  0.85 (mvp_c04..c07, c09):           ×5
- histórico con R²_real ≥ 0.85:                              ×20
- histórico con R²_real ∈ [0.70, 0.85):                      ×10
- resto:                                                     ×1

Splits estratificados: misma semilla 42, pero las 10 iter 3 se fuerzan al
split train (no contaminan val/test).

Salidas
-------
data/dataset/fields.npz
    Dataset extendido con sample_weight e is_iter3.
data/dataset/splits.npz
    Índices por split.
data/dataset/dataset_{train,val,test}.npz
    Subconjuntos por split (sobrescribe dataset vigente).
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.interpolate import interp1d


def resample_vector(vec, target_length, kind='cubic'):
    """
    Re-muestrear un vector 1D a longitud fija.

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
    """
    n = len(vec)
    if n == target_length:
        return vec.astype(np.float64)
    if n < 4:
        kind = 'linear'
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, vec, kind=kind, assume_sorted=True)
    return f(x_new)


def process_iter3_json(json_path, target_length):
    """
    Construir una muestra (X, amplitude, phase, r2, ny_original) desde un JSON iter 3.

    Parametros
    ----------
    json_path : str
        Ruta al JSON confirmado (iter 3).
    target_length : int
        Longitud objetivo para el re-muestreo.

    Devuelve
    --------
    dict
        Muestra con claves: sim_id, X, amplitude, phase, r_squared, ny_original.

    Errores
    -------
    ValueError
        Si widths no tiene longitud 101.
    """
    with open(json_path) as f:
        d = json.load(f)
    widths = np.asarray(d['widths'], dtype=np.float64)
    c2c = float(d['c2c_d']) if not isinstance(d['c2c_d'], list) else float(np.mean(d['c2c_d']))
    if len(widths) != 101:
        raise ValueError(f"{json_path}: widths n={len(widths)} != 101")
    trans = np.asarray(d['transmittance'], dtype=np.float64)
    phase = np.asarray(d['phase'], dtype=np.float64)
    ny = len(phase)
    trans_rs = np.clip(resample_vector(trans, target_length), 0.0, 1.0)
    phase_rs = resample_vector(phase, target_length)
    amplitude = np.sqrt(trans_rs)
    phase_centered = phase_rs - np.median(phase_rs)
    X = np.concatenate([widths, [c2c]])
    return {
        'sim_id': d.get('sim_id', os.path.basename(json_path).replace('_features.json', '')),
        'X': X.astype(np.float64),
        'amplitude': amplitude.astype(np.float64),
        'phase': phase_centered.astype(np.float64),
        'r_squared': float(d['r_squared']),
        'ny_original': int(ny),
    }


def compute_sample_weight(sim_id, r2, is_iter3):
    """
    Asignar sample_weight según el esquema v4.

    Parametros
    ----------
    sim_id : str
        Identificador de simulación.
    r2 : float
        R²_real asociado.
    is_iter3 : bool
        Indica si la simulación pertenece a iter 3.

    Devuelve
    --------
    float
        Peso asignado.
    """
    if is_iter3:
        return 30.0 if r2 >= 0.85 else 5.0
    if r2 >= 0.85:
        return 20.0
    if r2 >= 0.70:
        return 10.0
    return 1.0


def stratify_by_r2(r_squared):
    """
    Asignar banda por R²_paper.

    Parametros
    ----------
    r_squared : ndarray of float, shape (N,)
        R²_paper por simulación.

    Devuelve
    --------
    ndarray of int, shape (N,)
        Índice de banda (0..3).
    """
    bins = [-np.inf, 0.30, 0.70, 0.85, np.inf]
    return np.digitize(r_squared, bins) - 1


def split_stratified_forcing_train(r_squared, forced_train_mask, train_ratio, val_ratio, test_ratio, seed):
    """
    Generar splits estratificados forzando ciertas muestras al train.

    Parametros
    ----------
    r_squared : ndarray of float, shape (N,)
        R²_paper por simulación.
    forced_train_mask : ndarray of bool, shape (N,)
        True para muestras que deben caer en train.
    train_ratio : float
        Proporción objetivo de train.
    val_ratio : float
        Proporción objetivo de validation.
    test_ratio : float
        Proporción objetivo de test.
    seed : int
        Semilla para el shuffle estratificado.

    Devuelve
    --------
    tuple of ndarray
        (train_idx, val_idx, test_idx) con índices 0-based.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    bands = stratify_by_r2(r_squared)
    train_idx, val_idx, test_idx = [], [], []
    for band in np.unique(bands):
        band_mask = (bands == band)
        band_indices_all = np.where(band_mask)[0]
        # Separar forced y libres
        forced_in_band = [i for i in band_indices_all if forced_train_mask[i]]
        free_in_band = [i for i in band_indices_all if not forced_train_mask[i]]
        rng.shuffle(free_in_band)
        train_idx.extend(forced_in_band)
        n = len(free_in_band)
        n_train_free = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(free_in_band[:n_train_free])
        val_idx.extend(free_in_band[n_train_free:n_train_free + n_val])
        test_idx.extend(free_in_band[n_train_free + n_val:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), np.array(sorted(test_idx))


def report_split(name, r_squared, weights):
    """
    Imprimir distribución de bandas y pesos por split.

    Parametros
    ----------
    name : str
        Nombre del split (TRAIN/VAL/TEST).
    r_squared : ndarray of float, shape (N,)
        R²_paper del split.
    weights : ndarray of float, shape (N,)
        sample_weight del split.
    """
    bands = stratify_by_r2(r_squared)
    labels = ['R²<0.30', 'R²∈[0.30, 0.70)', 'R²∈[0.70, 0.85)', 'R²≥0.85']
    print(f"  {name}: N={len(r_squared)}, R²≥0.85={(r_squared >= 0.85).sum()}, "
          f"Σweights={weights.sum():.0f}, mean_w={weights.mean():.2f}, max_w={weights.max():.0f}")
    for i, lab in enumerate(labels):
        m = bands == i
        if m.any():
            print(f"    {lab:<22} n={int(m.sum()):4d}  Σw={float(weights[m].sum()):6.0f}")


def main():
    parser = argparse.ArgumentParser(description="Extiende fields.npz con iter 3 y sample_weight.")
    parser.add_argument('--input_fields', default='data/dataset/fields.npz',
                        help='fields.npz base generado por resample_focal_fields.py.')
    parser.add_argument('--iter3_dir', default='data/confirmed/iter3',
                        help='JSONs de los 10 candidatos confirmados de iter 3')
    parser.add_argument('--output_dir', default='data/dataset',
                        help='Dataset oficial extendido (sobrescribe la versión vigente).')
    parser.add_argument('--target_length', type=int, default=1024)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(" EXTENDIENDO DATASET CON ITER 3")
    print("=" * 80)

    # Cargar dataset base.
    base = np.load(args.input_fields, allow_pickle=True)
    sim_ids = list(base['sim_ids'])
    X = base['X']
    amplitude = base['amplitude']
    phase = base['phase']
    r_squared = base['r_squared']
    ny_original = base['ny_original'] if 'ny_original' in base.files else np.full(len(sim_ids), args.target_length)
    target_length = int(base['target_length'])
    if target_length != args.target_length:
        raise ValueError(f"target_length mismatch: base={target_length}, requested={args.target_length}")
    print(f"\n[1/4] Base cargada: {len(sim_ids)} sims, L={target_length}")

    # Cargar confirmaciones de iteracion 3.
    iter3_paths = sorted(glob.glob(os.path.join(args.iter3_dir, 'mvp_c*.json')))
    print(f"\n[2/4] Procesando {len(iter3_paths)} JSONs de iter 3...")
    iter3_samples = []
    for p in iter3_paths:
        s = process_iter3_json(p, target_length)
        iter3_samples.append(s)
        print(f"  {s['sim_id']:<10}  R²_real={s['r_squared']:.4f}  ny_orig={s['ny_original']}")

    # Concatenar muestras historicas y confirmadas.
    print(f"\n[3/4] Consolidando...")
    new_sim_ids = sim_ids + [s['sim_id'] for s in iter3_samples]
    new_X = np.vstack([X] + [s['X'][None, :] for s in iter3_samples])
    new_amp = np.vstack([amplitude] + [s['amplitude'][None, :] for s in iter3_samples])
    new_phase = np.vstack([phase] + [s['phase'][None, :] for s in iter3_samples])
    new_r2 = np.concatenate([r_squared, [s['r_squared'] for s in iter3_samples]])
    new_ny = np.concatenate([ny_original, [s['ny_original'] for s in iter3_samples]])
    is_iter3 = np.array([sid.startswith('mvp_c') for sid in new_sim_ids])

    # Calcular pesos por muestra.
    sample_weight = np.array([
        compute_sample_weight(sid, r2, it3)
        for sid, r2, it3 in zip(new_sim_ids, new_r2, is_iter3)
    ], dtype=np.float32)

    print(f"  N total: {len(new_sim_ids)} (= {len(sim_ids)} hist + {len(iter3_samples)} iter 3)")
    print(f"  Distribución de sample_weight:")
    for w in sorted(set(sample_weight.tolist())):
        n = int((sample_weight == w).sum())
        print(f"    weight={w:5.1f}  n={n:4d}")

    # Guardar campos extendidos.
    fields_out = os.path.join(args.output_dir, 'fields.npz')
    np.savez_compressed(fields_out,
                        sim_ids=np.array(new_sim_ids), X=new_X, amplitude=new_amp,
                        phase=new_phase, r_squared=new_r2, ny_original=new_ny,
                        sample_weight=sample_weight, is_iter3=is_iter3,
                        target_length=target_length)
    print(f"  {fields_out}  ({os.path.getsize(fields_out)/1e6:.1f} MB)")

    # Forzar nuevas muestras confirmadas dentro de train.
    print(f"\n[4/4] Splits estratificados (iter 3 forzados a train, seed={args.random_seed})...")
    train_idx, val_idx, test_idx = split_stratified_forcing_train(
        new_r2, is_iter3, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed
    )
    # Los splits deben ser disjuntos.
    assert not (set(train_idx) & set(val_idx)), "Train/Val overlap"
    assert not (set(train_idx) & set(test_idx)), "Train/Test overlap"
    assert not (set(val_idx) & set(test_idx)), "Val/Test overlap"
    # Todas las muestras de iteracion 3 deben permanecer en train.
    iter3_indices = np.where(is_iter3)[0]
    assert set(iter3_indices.tolist()).issubset(set(train_idx.tolist())), "Algún iter 3 no quedó en train"

    print()
    report_split("TRAIN", new_r2[train_idx], sample_weight[train_idx])
    print()
    report_split("VAL  ", new_r2[val_idx], sample_weight[val_idx])
    print()
    report_split("TEST ", new_r2[test_idx], sample_weight[test_idx])
    print()

    splits_path = os.path.join(args.output_dir, 'splits.npz')
    np.savez_compressed(splits_path, train=train_idx, val=val_idx, test=test_idx, seed=args.random_seed)
    print(f"  {splits_path}")

    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        out_path = os.path.join(args.output_dir, f'dataset_{name}.npz')
        np.savez_compressed(out_path,
                            sim_ids=np.array(new_sim_ids)[idx], X=new_X[idx],
                            amplitude=new_amp[idx], phase=new_phase[idx],
                            r_squared=new_r2[idx], ny_original=new_ny[idx],
                            sample_weight=sample_weight[idx],
                            is_iter3=is_iter3[idx],
                            target_length=target_length)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  dataset_{name}.npz  N={len(idx):4d}  ({size_mb:.1f} MB)")
    print()


if __name__ == '__main__':
    main()
