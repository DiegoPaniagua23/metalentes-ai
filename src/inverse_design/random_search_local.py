"""
Generar perturbaciones gaussianas locales alrededor de una geometria semilla.

Cada perturbación se valida contra las restricciones físicas:
  - widths ∈ [0.020, 0.300] µm
  - c2c_d ∈ [0.150, 0.300] µm
  - pared mínima de oro: c2c - max(widths) ≥ 0.020
  - duty cycle: 0.1 ≤ widths/c2c ≤ 0.5

Las perturbaciones se generan con clip al manifold válido (no rejection sampling):
primero se muestrea c2c y se calculan los rangos de widths derivados de las restricciones;
luego se muestrean los widths y se clipean a ese rango. Esto garantiza validez física
incluso cuando el seed está en el borde del manifold (p.ej. duty cycle exactamente 0.10
o 0.50). El output es un CSV en formato compatible con confirm_candidates.py.

Uso
---
python src/inverse_design/random_search_local.py \
    --seed_sim_id sim_042 \
    --n_perturbations 50 \
    --sigma_widths 0.005 \
    --sigma_c2c 0.005 \
    --output_csv results/inverse_design/top_candidates_hc042.csv

python src/inverse_design/random_search_local.py \
    --seed_json data/confirmed/iter4/mvp_c09_features.json \
    --n_perturbations 50 \
    --sigma_widths 0.005 --sigma_c2c 0.005 \
    --output_csv results/inverse_design/top_candidates_hc627.csv

Salidas
-------
CSV con columnas: cand_id;widths;c2c_d;r2_pred;fwhm_pred;concentration_pred;fitness;seed_origin
- cand_id: hc<tag>_<NNN>, donde <tag> es el sufijo del CSV (por defecto derivado del filename).
- r2_pred: R² conocido del seed (referencia, no es predicción de surrogate).
- fwhm_pred, conc_pred, fitness: 0.0 (campos dummy no usados por confirm_candidates.py).
- seed_origin: sim_id o cand_id del seed.

The first CSV row is the exact seed, followed by N perturbations.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


W_MIN, W_MAX = 0.020, 0.300       # µm
C2C_MIN, C2C_MAX = 0.150, 0.300   # µm
WALL_MIN = 0.020                  # pared mínima de oro = c2c - max(widths)
DUTY_MIN, DUTY_MAX = 0.10, 0.50


def is_valid(widths, c2c):
    """
    Verificar que una geometría cumple todas las restricciones físicas obligatorias.

    Parametros
    ----------
    widths : ndarray of float, shape (101,)
        Anchos de las rendijas en µm.
    c2c : float
        Distancia centro-a-centro en µm.

    Devuelve
    --------
    bool
        True si la geometría satisface rangos absolutos, pared mínima de Au y
        duty cycle; False si viola al menos una restricción.
    """
    if widths.min() < W_MIN or widths.max() > W_MAX:
        return False
    if c2c < C2C_MIN or c2c > C2C_MAX:
        return False
    if c2c - widths.max() < WALL_MIN:
        return False
    duty = widths / c2c
    if duty.min() < DUTY_MIN or duty.max() > DUTY_MAX:
        return False
    return True


def load_seed(args):
    """
    Cargar la geometría de un seed desde fields.npz o desde un JSON confirmado.

    Parametros
    ----------
    args : argparse.Namespace
        Debe contener exactamente uno de --seed_sim_id (lookup en fields.npz) o
        --seed_json (JSON con widths y c2c_d).

    Devuelve
    --------
    widths : ndarray of float, shape (101,)
        Anchos de las rendijas en µm.
    c2c : float
        Distancia centro-a-centro en µm (escalar; si el JSON lo trae como lista
        se promedia).
    r2 : float
        R²_real conocido del seed (referencia para el CSV de salida).
    seed_label : str
        Identificador del seed (sim_id o cand_id) usado como `seed_origin`.

    Errores
    -------
    SystemExit
        Si el seed no se encuentra, si widths no tiene longitud 101 o si no se
        especificó ninguna fuente.

    Notas
    -----
    Las semillas invalidas se reportan en stderr, pero la ejecucion continua.
    """
    if args.seed_sim_id:
        fields = np.load(args.fields_npz, allow_pickle=True)
        sim_ids = list(fields['sim_ids'])
        if args.seed_sim_id not in sim_ids:
            raise SystemExit(f"sim_id '{args.seed_sim_id}' no encontrado en {args.fields_npz}")
        idx = sim_ids.index(args.seed_sim_id)
        X = fields['X'][idx]
        widths = X[:101].astype(np.float64)
        c2c = float(X[101])
        r2 = float(fields['r_squared'][idx])
        seed_label = args.seed_sim_id
    elif args.seed_json:
        with open(args.seed_json) as f:
            d = json.load(f)
        widths = np.asarray(d['widths'], dtype=np.float64)
        c2c = d['c2c_d']
        if isinstance(c2c, list):
            c2c = float(np.mean(c2c))
        else:
            c2c = float(c2c)
        r2 = float(d.get('r_squared', 0.0))
        seed_label = d.get('cand_id') or d.get('sim_id') or os.path.basename(args.seed_json).replace('_features.json', '')
    else:
        raise SystemExit("Debes especificar --seed_sim_id o --seed_json")
    if len(widths) != 101:
        raise SystemExit(f"widths n={len(widths)} != 101")
    if not is_valid(widths, c2c):
        print(f"Seed {seed_label} NO cumple restricciones físicas. Se continúa de todos modos.",
              file=sys.stderr)
    return widths, c2c, r2, seed_label


def perturb(widths_seed, c2c_seed, sigma_w, sigma_c2c, rng, max_tries=20):
    """
    Perturbación gaussiana + clip al manifold válido.

    Estrategia: primero perturbar c2c (clip a [C2C_MIN, C2C_MAX]), luego perturbar
    widths con rangos derivados de c2c que GARANTIZAN restricciones físicas:
        w_min_allowed = max(W_MIN, DUTY_MIN · c2c)         (= max(0.020, 0.10·c2c))
        w_max_allowed = min(W_MAX, DUTY_MAX · c2c,
                            c2c - WALL_MIN)                (= min(0.300, 0.50·c2c, c2c-0.020))

    Devuelve
    --------
    tuple
        Perturbed widths and c2c, or (None, None) if no valid sample is found.
    """
    for _ in range(max_tries):
        # Perturbar c2c antes de derivar limites validos de ancho.
        c = c2c_seed + rng.normal(0, sigma_c2c)
        c = float(np.clip(c, C2C_MIN, C2C_MAX))
        w_min_allowed = max(W_MIN, DUTY_MIN * c)
        w_max_allowed = min(W_MAX, DUTY_MAX * c, c - WALL_MIN)
        if w_min_allowed >= w_max_allowed:
            continue
        # Recortar anchos al intervalo factible para c.
        w = widths_seed + rng.normal(0, sigma_w, size=101)
        w = np.clip(w, w_min_allowed, w_max_allowed)
        if is_valid(w, c):
            return w, c
    return None, None


def main():
    """
    Generar un CSV de perturbaciones para hill-climbing FDTD desde un seed.

    Escribe (n_perturbations + 1) filas: la primera es el seed exacto (sanity
    check al re-confirmar con FDTD), las restantes son perturbaciones gaussianas
    con σ_widths y σ_c2c configurables, clipeadas al manifold válido.

    Devuelve
    --------
    None
        Escribe --output_csv con columnas compatibles con confirm_candidates.py.
        Imprime un resumen por candidato con la desviación L∞ vs seed.
    """
    p = argparse.ArgumentParser(description="Genera perturbaciones gaussianas válidas alrededor de un seed.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--seed_sim_id', help='ID de simulación en fields.npz')
    src.add_argument('--seed_json', help='Path a JSON confirmado con widths + c2c_d')
    p.add_argument('--fields_npz', default='data/dataset/fields.npz')
    p.add_argument('--n_perturbations', type=int, default=50,
                   help='Número de perturbaciones gaussianas a generar (el seed exacto se agrega como fila #1).')
    p.add_argument('--sigma_widths', type=float, default=0.005, help='σ de widths en µm (default 5 nm).')
    p.add_argument('--sigma_c2c', type=float, default=0.005, help='σ de c2c en µm (default 5 nm).')
    p.add_argument('--seed', type=int, default=42, help='Semilla del RNG.')
    p.add_argument('--output_csv', required=True, help='CSV de salida (formato top_candidates).')
    p.add_argument('--tag', default=None,
                   help='Prefijo de cand_id (e.g. "hc042"). Si no se da, se deriva del filename.')
    args = p.parse_args()

    widths_seed, c2c_seed, r2_seed, seed_label = load_seed(args)
    rng = np.random.default_rng(args.seed)

    tag = args.tag or os.path.basename(args.output_csv).replace('top_candidates_', '').replace('.csv', '')

    print("=" * 80)
    print(" GENERADOR LOCAL DE PERTURBACIONES")
    print("=" * 80)
    print(f"  Seed: {seed_label}  (R²={r2_seed:.4f})")
    print(f"  σ_widths = {args.sigma_widths * 1000:.1f} nm,  σ_c2c = {args.sigma_c2c * 1000:.1f} nm")
    print(f"  N perturbaciones: {args.n_perturbations}  (+1 fila inicial con el seed exacto)")
    print(f"  Output: {args.output_csv}")
    print(f"  Tag: {tag}")
    print()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter=';')
        writer.writerow(['cand_id', 'widths', 'c2c_d', 'r2_pred', 'fwhm_pred',
                         'concentration_pred', 'fitness', 'seed_origin'])

        # Primera fila: semilla exacta.
        widths_str = ','.join(f'{w:.6f}' for w in widths_seed)
        cand_id_seed = f'{tag}_001'
        writer.writerow([cand_id_seed, widths_str, f'{c2c_seed:.6f}',
                         f'{r2_seed:.4f}', '0', '0', '0', seed_label])
        print(f"  [{cand_id_seed}]  seed exacto                 (R²_ref={r2_seed:.4f})")

        rejected = 0
        for i in range(2, args.n_perturbations + 2):
            w, c = perturb(widths_seed, c2c_seed, args.sigma_widths, args.sigma_c2c, rng)
            if w is None:
                rejected += 1
                continue
            widths_str = ','.join(f'{w_j:.6f}' for w_j in w)
            cand_id = f'{tag}_{i:03d}'
            linf_w = np.max(np.abs(w - widths_seed)) * 1000  # nm
            dc = abs(c - c2c_seed) * 1000
            writer.writerow([cand_id, widths_str, f'{c:.6f}',
                             f'{r2_seed:.4f}', '0', '0', '0', seed_label])
            print(f"  [{cand_id}]  L∞_w={linf_w:5.2f} nm, Δc2c={dc:4.2f} nm")

    print()
    print(f"  Generadas {args.n_perturbations + 1 - rejected} filas en {args.output_csv}")
    if rejected:
        print(f"  {rejected} perturbaciones rechazadas por restricciones.")
    print()


if __name__ == '__main__':
    main()
