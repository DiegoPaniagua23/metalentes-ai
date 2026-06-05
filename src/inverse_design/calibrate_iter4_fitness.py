"""
Recalibrar coeficientes del combo fitness desde multiples lotes confirmados.

Generaliza calibrate_iter3_fitness.py: acepta múltiples (top_csv, confirmed_dir) y los
empareja para construir el dataset de calibración OLS más amplio. La forma del modelo
permanece igual:

  R²_real = β0 + β_c·conc + β_f·fwhm

(R²_pred omitido por multicolinealidad cuando var<1e-6 en cada iteración; con n=20
se reintenta incluir si la var global cruza el umbral.)

Salidas
-------
results/inverse_design/combo_calibration.json
    {alpha, beta, ols_r2_fit, std_conc, std_fwhm, n_samples, data}; sobrescribe v4.

Notas
-----
Usar este script solo cuando las metricas de candidatos sean comparables entre lotes.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


def load_candidates(top_csv_path, confirmed_dir, label):
    """
    Emparejar top_candidates.csv con JSONs confirmados de una iteración.

    Parametros
    ----------
    top_csv_path : str
        Ruta a top_candidates.csv con outputs del surrogate (r2_pred, fwhm_pred, conc_pred).
    confirmed_dir : str
        Directorio con JSONs confirmados (`mvp_c*_features.json`) con `r_squared`.
    label : str
        Etiqueta de la iteración ('iter3', 'iter4', etc.) para trazabilidad en el OLS.

    Devuelve
    --------
    list of dict
        Cada dict con keys: iter, cand_id, r2_real, r2_pred, fwhm_pred, conc_pred.
    """
    with open(top_csv_path) as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    cand_pred = {r['cand_id']: r for r in rows}
    real = {}
    for fp in sorted(glob.glob(os.path.join(confirmed_dir, 'mvp_c*.json'))):
        with open(fp) as f:
            d = json.load(f)
        real[d['cand_id']] = float(d['r_squared'])
    paired = []
    for cid, r in cand_pred.items():
        if cid in real:
            paired.append({
                'iter': label,
                'cand_id': cid,
                'r2_real': real[cid],
                'r2_pred': float(r['r2_pred']),
                'fwhm_pred': float(r['fwhm_pred']),
                'conc_pred': float(r['concentration_pred']),
            })
    return paired


def ols_fit(y, X):
    """
    Ajustar un modelo OLS con intercepto.

    Parametros
    ----------
    y : ndarray of float, shape (N,)
        Variable objetivo.
    X : ndarray of float, shape (N, d)
        Variables explicativas (sin columna de unos).

    Devuelve
    --------
    tuple
        (intercept, coefs, r2); intercept es escalar, coefs shape (d,), r2 del ajuste.
    """
    X_design = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ coefs
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coefs[0]), coefs[1:], r2


def main():
    """
    Calibrar coeficientes del combo fitness desde pares candidato/confirmacion.

    Devuelve
    --------
    None
        Escribe combo_calibration.json en --output_dir con α, β, coeficientes OLS,
        R²_OLS, std de conc y fwhm, y los pares (cand_id, r2_real, r2_pred, ...)
        usados en el ajuste para trazabilidad.
    """
    parser = argparse.ArgumentParser(description="Calibra coeficientes del combo fitness con OLS.")
    parser.add_argument('--iter3_csv', required=True)
    parser.add_argument('--iter3_dir', default='data/confirmed/iter3')
    parser.add_argument('--iter4_csv', required=True)
    parser.add_argument('--iter4_dir', default='data/confirmed/iter4')
    parser.add_argument('--output_dir', default='results/inverse_design')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(" CALIBRACIÓN COMBO FITNESS: OLS sobre iter 3 + iter 4 (n=20)")
    print("=" * 80)

    paired = []
    paired += load_candidates(args.iter3_csv, args.iter3_dir, 'iter3')
    paired += load_candidates(args.iter4_csv, args.iter4_dir, 'iter4')
    print(f"\n  Datos: {len(paired)} candidatos emparejados (iter3 + iter4)")
    print()
    print(f"  {'iter':<6}{'cand':<8}{'R²_real':>10}{'R²_pred':>10}{'fwhm':>10}{'conc':>10}")
    for p in paired:
        print(f"  {p['iter']:<6}{p['cand_id']:<8}{p['r2_real']:>10.4f}{p['r2_pred']:>10.4f}{p['fwhm_pred']:>10.2f}{p['conc_pred']:>10.4f}")

    y = np.array([p['r2_real'] for p in paired])
    r2_pred = np.array([p['r2_pred'] for p in paired])
    fwhm = np.array([p['fwhm_pred'] for p in paired])
    conc = np.array([p['conc_pred'] for p in paired])

    var_r2_pred = float(r2_pred.var())
    var_fwhm = float(fwhm.var())
    var_conc = float(conc.var())
    print()
    print(f"  Varianzas:  R²_pred={var_r2_pred:.6f}  fwhm={var_fwhm:.4f}  conc={var_conc:.6f}")
    print(f"  Std:        R²_pred={r2_pred.std():.6f}  fwhm={fwhm.std():.4f}  conc={conc.std():.6f}")

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    print()
    print(f"  corr(R²_real, R²_pred)   = {corr(y, r2_pred):+.3f}")
    print(f"  corr(R²_real, fwhm_pred) = {corr(y, fwhm):+.3f}")
    print(f"  corr(R²_real, conc_pred) = {corr(y, conc):+.3f}")

    use_r2_pred_in_ols = var_r2_pred > 1e-6
    if use_r2_pred_in_ols:
        X = np.column_stack([r2_pred, conc, fwhm])
        intercept, coefs, r2_ols = ols_fit(y, X)
        beta_r2, beta_c, beta_f = coefs[0], coefs[1], coefs[2]
        print()
        print(f"  OLS modelo (con R²_pred): R²_real = {intercept:.4f} + {beta_r2:.4f}·R²_pred + {beta_c:.4f}·conc + {beta_f:.4f}·fwhm")
    else:
        X = np.column_stack([conc, fwhm])
        intercept, coefs, r2_ols = ols_fit(y, X)
        beta_r2 = 0.0
        beta_c, beta_f = coefs[0], coefs[1]
        print()
        print(f"  R²_pred omitido (var<1e-6). OLS modelo: R²_real = {intercept:.4f} + {beta_c:.4f}·conc + {beta_f:.4f}·fwhm")
    print(f"  R²_OLS = {r2_ols:.4f}")

    alpha = float(beta_c * conc.std())
    beta = float(-beta_f * fwhm.std())

    print()
    print(f"  Coeficientes para combo fitness (z-scores normalizados):")
    print(f"    α (peso conc_z)  = {alpha:+.4f}")
    print(f"    β (peso fwhm_z)  = {beta:+.4f}   (se RESTA en fitness)")
    print(f"    fitness = R²_pred + {alpha:+.4f}·conc_z − {beta:+.4f}·fwhm_z")

    out_path = os.path.join(args.output_dir, 'combo_calibration.json')
    with open(out_path, 'w') as f:
        json.dump({
            'alpha': alpha, 'beta': beta,
            'ols_intercept': intercept,
            'ols_beta_r2': beta_r2,
            'ols_beta_conc': float(beta_c),
            'ols_beta_fwhm': float(beta_f),
            'ols_r2_fit': r2_ols,
            'used_r2_pred_in_ols': use_r2_pred_in_ols,
            'std_conc': float(conc.std()),
            'std_fwhm': float(fwhm.std()),
            'mean_r2_pred': float(r2_pred.mean()),
            'n_samples': len(paired),
            'data': paired,
            'description': 'fitness = R²_pred + α·conc_z − β·fwhm_z (combo iter 5: iter3+iter4, n=20)',
        }, f, indent=2)
    print(f"\n  Calibración guardada: {out_path}")
    print()


if __name__ == '__main__':
    main()
