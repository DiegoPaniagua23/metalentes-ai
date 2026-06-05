"""
Construir splits estratificados de entrenamiento, validacion y prueba.

Lee data/dataset/fields.npz (output de resample_focal_fields.py) y genera
splits 70/15/15 estratificados por R²_paper (referencia, NO target).

Salidas
-------
data/dataset/splits.npz
    Índices por split + metadatos.
data/dataset/dataset_train.npz, dataset_val.npz, dataset_test.npz
    Subconjuntos con campos, fase y geometría.

Notas
-----
La estratificación preserva proporciones por banda:
- R²_paper < 0.30 (mayoría)
- R²_paper ∈ [0.30, 0.70) (medio)
- R²_paper ∈ [0.70, 0.85) (alto-medio)
- R²_paper ≥ 0.85 (cola superior)
"""

import argparse
import os

import numpy as np


def stratify_by_r2(r_squared: np.ndarray, n_bands: int = 4) -> np.ndarray:
    """
    Asignar a cada muestra una banda según R²_paper.

    Parametros
    ----------
    r_squared : ndarray of float, shape (N,)
        R²_paper por simulación.
    n_bands : int, optional
        Número de bandas esperadas (default 4).

    Devuelve
    --------
    ndarray of int, shape (N,)
        Índice de banda (0..3).
    """
    bins = [-np.inf, 0.30, 0.70, 0.85, np.inf]
    return np.digitize(r_squared, bins) - 1  # 0..3


def split_stratified(r_squared: np.ndarray, train_ratio: float, val_ratio: float,
                      test_ratio: float, seed: int) -> tuple:
    """
    Generar índices estratificados para train/val/test.

    Parametros
    ----------
    r_squared : ndarray of float, shape (N,)
        R²_paper por simulación.
    train_ratio : float
        Proporción de train.
    val_ratio : float
        Proporción de validation.
    test_ratio : float
        Proporción de test.
    seed : int
        Semilla para el shuffle estratificado.

    Devuelve
    --------
    tuple of ndarray
        (train_idx, val_idx, test_idx) con índices 0-based.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios deben sumar 1"
    rng = np.random.default_rng(seed)
    bands = stratify_by_r2(r_squared)
    train_idx, val_idx, test_idx = [], [], []
    for band in np.unique(bands):
        band_mask = (bands == band)
        band_indices = np.where(band_mask)[0]
        rng.shuffle(band_indices)
        n = len(band_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(band_indices[:n_train])
        val_idx.extend(band_indices[n_train:n_train + n_val])
        test_idx.extend(band_indices[n_train + n_val:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), np.array(sorted(test_idx))


def report_split(name: str, r_squared: np.ndarray):
    """
    Imprimir distribución de bandas por split.

    Parametros
    ----------
    name : str
        Nombre del split (TRAIN/VAL/TEST).
    r_squared : ndarray of float, shape (N,)
        R²_paper del split.
    """
    bands = stratify_by_r2(r_squared)
    labels = ['R²<0.30', 'R²∈[0.30, 0.70)', 'R²∈[0.70, 0.85)', 'R²≥0.85']
    print(f"  {name}: total={len(r_squared)}, R²≥0.90={sum(r_squared >= 0.90)}")
    for i, lab in enumerate(labels):
        count = int(np.sum(bands == i))
        pct = 100 * count / len(r_squared) if len(r_squared) > 0 else 0
        print(f"    {lab:<25} {count:4d}  ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Genera splits estratificados train/val/test.")
    parser.add_argument('--input', default='data/dataset/fields.npz', help='Input fields.npz')
    parser.add_argument('--output_dir', default='data/dataset', help='Directorio de salida')
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(" Splits Estratificados ")
    print("=" * 80)

    data = np.load(args.input, allow_pickle=True)
    sim_ids = data['sim_ids']
    X = data['X']
    amplitude = data['amplitude']
    phase = data['phase']
    r_squared = data['r_squared']
    target_length = int(data['target_length'])
    # La longitud original se usa para la resolucion efectiva de propagacion.
    ny_original = data['ny_original'] if 'ny_original' in data.files else np.full(len(sim_ids), target_length)

    N = len(sim_ids)
    print(f"\n  Cargado: {args.input}")
    print(f"  N total: {N}, target_length: {target_length}")
    print(f"  R²≥0.90: {sum(r_squared >= 0.90)}")
    print()

    # Generar indices estratificados.
    train_idx, val_idx, test_idx = split_stratified(
        r_squared, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed
    )

    print("Distribución por split (estratificada por R²):")
    report_split("TRAIN", r_squared[train_idx])
    print()
    report_split("VAL  ", r_squared[val_idx])
    print()
    report_split("TEST ", r_squared[test_idx])
    print()

    # Los splits deben ser exhaustivos y disjuntos.
    assert len(set(train_idx) | set(val_idx) | set(test_idx)) == N, "Splits no cubren todos los samples"
    assert not (set(train_idx) & set(val_idx)), "Train/Val overlap"
    assert not (set(train_idx) & set(test_idx)), "Train/Test overlap"
    assert not (set(val_idx) & set(test_idx)), "Val/Test overlap"

    # Guardar indices de splits y datasets materializados.
    splits_path = os.path.join(args.output_dir, 'splits.npz')
    np.savez_compressed(splits_path, train=train_idx, val=val_idx, test=test_idx, seed=args.random_seed)
    print(f"Splits guardados: {splits_path}")

    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        out_path = os.path.join(args.output_dir, f'dataset_{name}.npz')
        np.savez_compressed(out_path,
                            sim_ids=sim_ids[idx], X=X[idx],
                            amplitude=amplitude[idx], phase=phase[idx],
                            r_squared=r_squared[idx],
                            ny_original=ny_original[idx],
                            target_length=target_length)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"dataset_{name}.npz  N={len(idx):4d}  ({size_mb:.1f} MB)")
    print()


if __name__ == '__main__':
    main()
