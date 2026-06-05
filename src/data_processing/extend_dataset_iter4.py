"""
Extend the dataset with iteration 4 confirmations.

Toma el fields.npz vigente (1100 sims = 1090 hist + 10 iter 3, ya con sample_weight)
y agrega los 10 confirmados de data/confirmed/iter4/. Los sim_ids de iter 4 se
prefijan con "iter4_" para evitar colisión con los iter 3 (ambos usan mvp_cXX).

sample_weight asignados a los 10 iter 4:
- R²_real ≥ 0.90 (mvp_c09):                 ×40   (top tier: dato más raro del dataset)
- R²_real ∈ [0.85, 0.90) (5: c01, c04..06, c10): ×30
- R²_real <  0.85 (4: c02, c03, c07, c08):  ×5

Pesos de iter 3 e histórico se PRESERVAN tal como vienen en el npz vigente (no se
recalculan). Esto mantiene la jerarquía v4: histórico R²≥0.85 ×20, [0.70,0.85) ×10,
iter3 R²≥0.85 ×30, iter3 R²<0.85 ×5, resto ×1.

Splits v5: iter 3 + iter 4 forzados a train (no contaminan val/test). Las muestras
históricas conservan los splits originales (mismo random_seed=42, banda por R²).

Salidas
-------
data/dataset/fields.npz
    Dataset extendido a 1110 sims con sample_weight v5 e is_iter3 + is_iter4.
data/dataset/splits.npz
data/dataset/dataset_{train,val,test}.npz
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.interpolate import interp1d


def resample_vector(vec, target_length, kind='cubic'):
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


def process_iter_json(json_path, target_length, sim_id_prefix=""):
    """
    Construir una muestra (X, amplitude, phase, r2, ny_original) desde un JSON confirmado.

    Parametros
    ----------
    json_path : str
        Ruta al JSON confirmado (iter N).
    target_length : int
        Longitud objetivo para el re-muestreo.
    sim_id_prefix : str, optional
        Prefijo a anteponer al cand_id para evitar colisión de sim_ids entre
        iteraciones (ej. "iter4_" produce "iter4_mvp_c01").

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
    cand_id = d.get('cand_id') or os.path.basename(json_path).replace('_features.json', '')
    sim_id = sim_id_prefix + cand_id
    X = np.concatenate([widths, [c2c]])
    return {
        'sim_id': sim_id,
        'X': X.astype(np.float64),
        'amplitude': amplitude.astype(np.float64),
        'phase': phase_centered.astype(np.float64),
        'r_squared': float(d['r_squared']),
        'ny_original': int(ny),
    }


def compute_iter4_weight(r2):
    """
    Asignar sample_weight a una muestra iter 4 según su R²_real.

    Parametros
    ----------
    r2 : float
        R²_real asociado.

    Devuelve
    --------
    float
        Peso (×40 si R²≥0.90, ×30 si R²≥0.85, ×5 en otro caso).

    Notas
    -----
    Higher R2 bands receive larger sample weights.
    """
    if r2 >= 0.90:
        return 40.0
    if r2 >= 0.85:
        return 30.0
    return 5.0


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
        Índice de banda (0..3): <0.30, [0.30,0.70), [0.70,0.85), ≥0.85.
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
        True para muestras que deben caer en train (iter 3 + iter 4).
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
    parser = argparse.ArgumentParser(description="Extiende fields.npz con iter 4 y sample_weight.")
    parser.add_argument('--input_fields', default='data/dataset/fields.npz',
                        help='fields.npz vigente (debe contener ya iter 3, 1100 sims).')
    parser.add_argument('--iter4_dir', default='data/confirmed/iter4',
                        help='JSONs de los 10 candidatos confirmados de iter 4')
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
    print(" EXTENDIENDO DATASET CON ITER 4")
    print("=" * 80)

    # Cargar dataset vigente.
    base = np.load(args.input_fields, allow_pickle=True)
    sim_ids = list(base['sim_ids'])
    X = base['X']
    amplitude = base['amplitude']
    phase = base['phase']
    r_squared = base['r_squared']
    ny_original = base['ny_original'] if 'ny_original' in base.files else np.full(len(sim_ids), args.target_length)
    sample_weight_base = base['sample_weight'] if 'sample_weight' in base.files else np.ones(len(sim_ids), dtype=np.float32)
    is_iter3_base = base['is_iter3'] if 'is_iter3' in base.files else np.zeros(len(sim_ids), dtype=bool)
    target_length = int(base['target_length'])
    if target_length != args.target_length:
        raise ValueError(f"target_length mismatch: base={target_length}, requested={args.target_length}")
    print(f"\n[1/4] Base cargada: {len(sim_ids)} sims, L={target_length}")
    print(f"      sample_weight base: min={sample_weight_base.min():.1f} max={sample_weight_base.max():.1f}")
    print(f"      iter3 marcados: {int(is_iter3_base.sum())}")

    # Cargar confirmaciones de iteracion 4.
    iter4_paths = sorted(glob.glob(os.path.join(args.iter4_dir, 'mvp_c*.json')))
    if len(iter4_paths) == 0:
        raise SystemExit(f"No se encontraron JSONs en {args.iter4_dir}")
    print(f"\n[2/4] Procesando {len(iter4_paths)} JSONs de iter 4...")
    iter4_samples = []
    for p in iter4_paths:
        s = process_iter_json(p, target_length, sim_id_prefix="iter4_")
        iter4_samples.append(s)
        w = compute_iter4_weight(s['r_squared'])
        print(f"  {s['sim_id']:<18}  R²_real={s['r_squared']:.4f}  ny_orig={s['ny_original']}  weight=×{w:.0f}")

    # Concatenar muestras base y confirmadas.
    print(f"\n[3/4] Consolidando...")
    new_sim_ids = sim_ids + [s['sim_id'] for s in iter4_samples]
    new_X = np.vstack([X] + [s['X'][None, :] for s in iter4_samples])
    new_amp = np.vstack([amplitude] + [s['amplitude'][None, :] for s in iter4_samples])
    new_phase = np.vstack([phase] + [s['phase'][None, :] for s in iter4_samples])
    new_r2 = np.concatenate([r_squared, [s['r_squared'] for s in iter4_samples]])
    new_ny = np.concatenate([ny_original, [s['ny_original'] for s in iter4_samples]])
    is_iter3 = np.concatenate([is_iter3_base, np.zeros(len(iter4_samples), dtype=bool)])
    is_iter4 = np.array([sid.startswith('iter4_') for sid in new_sim_ids])

    iter4_weights = np.array([compute_iter4_weight(s['r_squared']) for s in iter4_samples], dtype=np.float32)
    sample_weight = np.concatenate([sample_weight_base.astype(np.float32), iter4_weights])

    forced_train_mask = is_iter3 | is_iter4

    print(f"  N total: {len(new_sim_ids)} (= {len(sim_ids)} base + {len(iter4_samples)} iter 4)")
    print(f"  Distribución de sample_weight:")
    for w in sorted(set(sample_weight.tolist())):
        n = int((sample_weight == w).sum())
        print(f"    weight={w:5.1f}  n={n:4d}")
    print(f"  Forzados a train: {int(forced_train_mask.sum())} (iter3={int(is_iter3.sum())} + iter4={int(is_iter4.sum())})")

    # Guardar campos extendidos.
    fields_out = os.path.join(args.output_dir, 'fields.npz')
    np.savez_compressed(fields_out,
                        sim_ids=np.array(new_sim_ids), X=new_X, amplitude=new_amp,
                        phase=new_phase, r_squared=new_r2, ny_original=new_ny,
                        sample_weight=sample_weight, is_iter3=is_iter3, is_iter4=is_iter4,
                        target_length=target_length)
    print(f"  {fields_out}  ({os.path.getsize(fields_out)/1e6:.1f} MB)")

    # Forzar muestras confirmadas dentro de train.
    print(f"\n[4/4] Splits estratificados (iter 3+4 forzados a train, seed={args.random_seed})...")
    train_idx, val_idx, test_idx = split_stratified_forcing_train(
        new_r2, forced_train_mask, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed
    )
    assert not (set(train_idx) & set(val_idx)), "Train/Val overlap"
    assert not (set(train_idx) & set(test_idx)), "Train/Test overlap"
    assert not (set(val_idx) & set(test_idx)), "Val/Test overlap"
    forced_indices = np.where(forced_train_mask)[0]
    assert set(forced_indices.tolist()).issubset(set(train_idx.tolist())), "Algún iter3/4 no quedó en train"

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
                            is_iter3=is_iter3[idx], is_iter4=is_iter4[idx],
                            target_length=target_length)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  dataset_{name}.npz  N={len(idx):4d}  ({size_mb:.1f} MB)")
    print()


if __name__ == '__main__':
    main()
