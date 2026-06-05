"""
Optimizar geometrías de metalentes con un algoritmo genetico.

Optimiza geometrías de metalente usando un fitness derivado del surrogate:
- r2_pred (iter 1-3)
- combo = R2_pred + alpha * conc_z - beta * fwhm_z (iter 4+)

Restricciones físicas obligatorias:
- widths_i ∈ [0.020, 0.300] µm
- c2c_d ∈ [0.150, 0.300] µm
- Pared mínima de oro: c2c_d - 0.5(w_j + w_{j+1}) ≥ 0.020 µm
- Duty cycle: 0.1 ≤ widths_i / c2c_d ≤ 0.5

Salidas
-------
results/inverse_design/top_candidates.csv
    Top-K geometrías candidatas.
results/inverse_design/ga_convergence.json
    Historial de métricas por generación.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: pytorch no instalado. Actualiza el entorno (consultoria_env.yml).")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'surrogate'))
from train_field_surrogate import FieldMLP, FieldCNN, apply_input_scaler


W_MIN, W_MAX = 0.020, 0.300
C_MIN, C_MAX = 0.150, 0.300
WALL_MIN = 0.020
DUTY_MIN, DUTY_MAX = 0.1, 0.5


def clip_to_physical(individual: np.ndarray) -> np.ndarray:
    """
    Reparar un individuo para cumplir restricciones físicas.

    Parametros
    ----------
    individual : ndarray of float, shape (102,)
        Vector con widths (101) y c2c_d (1).

    Devuelve
    --------
    ndarray of float, shape (102,)
        Individuo reparado dentro de límites físicos.

    Notas
    -----
    Layout: individual[:101] = widths, individual[101] = c2c_d.
    """
    ind = individual.copy()
    widths, c2c = ind[:101], ind[101]
    widths = np.clip(widths, W_MIN, W_MAX)
    c2c = np.clip(c2c, C_MIN, C_MAX)
    # Aplicar limites de duty-cycle antes de revisar paredes por pares.
    widths = np.clip(widths, DUTY_MIN * c2c, DUTY_MAX * c2c)
    # La proyeccion iterativa elimina violaciones locales de pared.
    for _ in range(3):
        max_allowed_pair = 2 * (c2c - WALL_MIN)
        for i in range(100):
            pair_sum = widths[i] + widths[i + 1]
            if pair_sum > max_allowed_pair:
                scale = max_allowed_pair / max(pair_sum, 1e-9)
                widths[i] *= scale
                widths[i + 1] *= scale
        widths = np.clip(widths, max(W_MIN, DUTY_MIN * c2c), min(W_MAX, DUTY_MAX * c2c))
    ind[:101] = widths
    ind[101] = c2c
    return ind


def is_valid(individual: np.ndarray) -> bool:
    """
    Verificar restricciones físicas sin reparar.

    Parametros
    ----------
    individual : ndarray of float, shape (102,)
        Vector con widths (101) y c2c_d (1).

    Devuelve
    --------
    bool
        True si cumple todas las restricciones.
    """
    widths, c2c = individual[:101], individual[101]
    if (widths.min() < W_MIN - 1e-9) or (widths.max() > W_MAX + 1e-9): return False
    if (c2c < C_MIN - 1e-9) or (c2c > C_MAX + 1e-9): return False
    duties = widths / c2c
    if (duties.min() < DUTY_MIN - 1e-9) or (duties.max() > DUTY_MAX + 1e-9): return False
    for i in range(100):
        if c2c - 0.5 * (widths[i] + widths[i + 1]) < WALL_MIN - 1e-9:
            return False
    return True


def random_individual(rng: np.random.Generator) -> np.ndarray:
    """
    Muestrear un individuo aleatorio dentro del espacio físico.

    Parametros
    ----------
    rng : np.random.Generator
        Generador aleatorio.

    Devuelve
    --------
    ndarray of float, shape (102,)
        Individuo válido (reparado).
    """
    c2c = rng.uniform(C_MIN, C_MAX)
    # Muestrear anchos dentro del intervalo factible para el c2c elegido.
    w_lo = max(W_MIN, DUTY_MIN * c2c)
    w_hi = min(W_MAX, DUTY_MAX * c2c, 2 * (c2c - WALL_MIN) - W_MIN)
    widths = rng.uniform(w_lo, w_hi, size=101)
    ind = np.concatenate([widths, [c2c]])
    return clip_to_physical(ind)


def initialize_population(pop_size: int, rng: np.random.Generator, seeds: np.ndarray = None,
                          seed_fraction: float = 0.2, seed_sigma: float = 0.005,
                          manifold_restricted: bool = False):
    """
    Inicializar población con opción de seeding.

    Parametros
    ----------
    pop_size : int
        Tamaño de la población.
    rng : np.random.Generator
        Generador aleatorio.
    seeds : ndarray or None
        Seeds de geometría, shape (n_seeds, 102).
    seed_fraction : float, optional
        Fracción de población inicializada a partir de seeds.
    seed_sigma : float, optional
        Sigma de perturbación gaussiana (µm).
    manifold_restricted : bool, optional
        Si True, toda la población es perturbación de seeds.

    Devuelve
    --------
    tuple
        (pop, origin) donde origin es None en modo estándar.
    """
    if manifold_restricted and seeds is not None and len(seeds) > 0:
        pop = np.zeros((pop_size, 102))
        origin = np.zeros(pop_size, dtype=int)
        for i in range(pop_size):
            seed_idx = i % len(seeds)
            base = seeds[seed_idx].copy()
            base[:101] += rng.normal(0, seed_sigma, size=101)
            base[101] += rng.normal(0, seed_sigma * 0.5)
            pop[i] = clip_to_physical(base)
            origin[i] = seed_idx
        return pop, origin
    pop = np.array([random_individual(rng) for _ in range(pop_size)])
    if seeds is not None and len(seeds) > 0:
        n_seeded = int(pop_size * seed_fraction)
        for i in range(n_seeded):
            base = seeds[rng.integers(len(seeds))].copy()
            base[:101] += rng.normal(0, seed_sigma, size=101)
            base[101] += rng.normal(0, seed_sigma * 0.5)
            pop[i] = clip_to_physical(base)
    return pop, None


def project_to_manifold(individual: np.ndarray, seed: np.ndarray, max_dev: float) -> np.ndarray:
    """
    Proyectar un individuo al box L_inf alrededor de un seed.

    Parametros
    ----------
    individual : ndarray of float, shape (102,)
        Individuo a proyectar.
    seed : ndarray of float, shape (102,)
        Seed de referencia.
    max_dev : float
        Desviación máxima L_inf permitida (µm).

    Devuelve
    --------
    ndarray of float, shape (102,)
        Individuo proyectado.
    """
    lo = seed - max_dev
    hi = seed + max_dev
    return np.clip(individual, lo, hi)


def tournament_select(pop: np.ndarray, fitness: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Seleccionar un individuo por torneo.

    Parametros
    ----------
    pop : ndarray, shape (N, 102)
        Población.
    fitness : ndarray, shape (N,)
        Fitness por individuo.
    k : int
        Tamaño del torneo.
    rng : np.random.Generator
        Generador aleatorio.

    Devuelve
    --------
    ndarray, shape (102,)
        Individuo seleccionado.
    """
    idx = rng.integers(len(pop), size=k)
    winner = idx[np.argmax(fitness[idx])]
    return pop[winner].copy()


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float, rng: np.random.Generator):
    """
    Aplicar crossover SBX (Simulated Binary Crossover).

    Parametros
    ----------
    p1 : ndarray, shape (102,)
        Padre 1.
    p2 : ndarray, shape (102,)
        Padre 2.
    eta : float
        Parámetro de distribución SBX.
    rng : np.random.Generator
        Generador aleatorio.

    Devuelve
    --------
    tuple of ndarray
        (child1, child2).
    """
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() < 0.5:
            if abs(p1[i] - p2[i]) > 1e-12:
                y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                u = rng.random()
                beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                c1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                c2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
    return c1, c2


def gaussian_mutation(ind: np.ndarray, rate: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Aplicar mutación gaussiana por gen con probabilidad rate.

    Parametros
    ----------
    ind : ndarray, shape (102,)
        Individuo a mutar.
    rate : float
        Probabilidad de mutación por gen.
    sigma : float
        Sigma de la perturbación gaussiana.
    rng : np.random.Generator
        Generador aleatorio.

    Devuelve
    --------
    ndarray, shape (102,)
        Individuo mutado.
    """
    out = ind.copy()
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] += rng.normal(0, sigma)
    return out


def compute_fitness_batch(individuals: np.ndarray, model, x_scaler, L: int, device,
                          fitness_mode: str = 'r2_pred', alpha: float = 0.0, beta: float = 0.0):
    """
    Computar fitness y métricas auxiliares del surrogate.

    Parametros
    ----------
    individuals : ndarray, shape (N, 102)
        Población a evaluar.
    model : torch.nn.Module
        Surrogate multi-output (campo + R²_pred).
    x_scaler : dict
        Escalador de inputs (train_field_surrogate).
    L : int
        Longitud del vector de campo.
    device : torch.device
        Dispositivo de ejecución.
    fitness_mode : {'r2_pred', 'combo'}, optional
        r2_pred usa R²_pred directo; combo agrega conc_z y fwhm_z.
    alpha : float, optional
        Peso de conc_z en modo combo.
    beta : float, optional
        Peso de fwhm_z en modo combo.

    Devuelve
    --------
    tuple
        (fitness, fwhm_vals, conc_vals, r2_pred) como ndarrays.

    Notas
    -----
    α y β provienen de calibrate_iter3_fitness.py (OLS sobre 10 iter 3).
    """
    X_n = apply_input_scaler(individuals, x_scaler).astype(np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(X_n).to(device)
        field_pred, r2_pred_tensor = model(x_t)
        if r2_pred_tensor is None:
            raise RuntimeError("El surrogate cargado no tiene cabeza R². Re-entrenar con --weight_r2 > 0.")
        r2_pred = r2_pred_tensor.cpu().numpy()
        amp_pred = field_pred.cpu().numpy()[:, :L]

    # Metricas auxiliares usadas para logs y fitness combo.
    int_pred = np.maximum(amp_pred, 0) ** 2
    fwhm_vals = np.zeros(len(individuals))
    conc_vals = np.zeros(len(individuals))
    for i, intensity in enumerate(int_pred):
        if intensity.max() <= 0:
            fwhm_vals[i] = L; conc_vals[i] = 0; continue
        half = intensity.max() / 2
        above = intensity >= half
        fwhm_vals[i] = float(np.where(above)[0][-1] - np.where(above)[0][0] + 1) if above.any() else L
        conc_vals[i] = float(intensity.max() / (intensity.mean() + 1e-12))

    if fitness_mode == 'combo':
        # Los z-scores poblacionales preservan el ranking local.
        fwhm_std = fwhm_vals.std() + 1e-9
        conc_std = conc_vals.std() + 1e-9
        fwhm_z = (fwhm_vals - fwhm_vals.mean()) / fwhm_std
        conc_z = (conc_vals - conc_vals.mean()) / conc_std
        fitness = r2_pred + alpha * conc_z - beta * fwhm_z
    else:
        fitness = r2_pred
    return fitness, fwhm_vals, conc_vals, r2_pred


def main():
    parser = argparse.ArgumentParser(description="GA para optimizar geometria de metalente.")
    parser.add_argument('--model_dir', default='results/surrogate')
    parser.add_argument('--output_dir', default='results/inverse_design')
    parser.add_argument('--pop_size', type=int, default=500)
    parser.add_argument('--n_gen', type=int, default=200)
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--tournament_k', type=int, default=5)
    parser.add_argument('--crossover_rate', type=float, default=0.80)
    parser.add_argument('--mutation_rate', type=float, default=0.05)
    parser.add_argument('--mutation_sigma', type=float, default=0.01)
    parser.add_argument('--sbx_eta', type=float, default=15.0)
    parser.add_argument('--elite_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_from_dataset', action='store_true', help='Sembrar con top geometrías del train set')
    parser.add_argument('--dataset_dir', default='data/dataset', help='Para seeding')
    parser.add_argument('--seed_top_k', type=int, default=20, help='Top-K reales como semillas (default 20)')
    parser.add_argument('--seed_fraction', type=float, default=0.20, help='Fracción de pop_size inicializada con seeds (default 0.20)')
    parser.add_argument('--seed_sigma', type=float, default=0.005, help='Sigma de perturbación gaussiana de los seeds (µm)')
    parser.add_argument('--manifold_restricted', action='store_true',
                        help='Restringe búsqueda a perturbaciones cercanas a los top-K reales. '
                             'Desactiva crossover global y limita deriva. Iteración 3+.')
    parser.add_argument('--max_deviation', type=float, default=0.020,
                        help='Desviación L_inf máxima permitida desde el seed original (µm). Solo con --manifold_restricted.')
    parser.add_argument('--multi_seed_from_jsons', default=None,
                        help='Directorio con JSONs cuya geometría (widths+c2c_d) sirve como seeds. '
                             'Reemplaza --seed_from_dataset. Activa manifold_restricted automáticamente.')
    parser.add_argument('--seed_min_r2', type=float, default=0.0,
                        help='Filtra los JSONs de --multi_seed_from_jsons aceptando solo r_squared ≥ umbral. '
                             '0.0 = sin filtro (todos).')
    parser.add_argument('--fitness_mode', default='r2_pred', choices=['r2_pred', 'combo'],
                        help='r2_pred o combo = R²_pred + alpha*conc_z - beta*fwhm_z.')
    parser.add_argument('--combo_calibration', default=None,
                        help='Path a combo_calibration.json (output de calibrate_iter3_fitness.py). '
                             'Solo aplica si --fitness_mode combo.')
    parser.add_argument('--combo_alpha', type=float, default=None,
                        help='Override de α (peso conc_z) sin pasar --combo_calibration.')
    parser.add_argument('--combo_beta', type=float, default=None,
                        help='Override de β (peso fwhm_z) sin pasar --combo_calibration.')
    parser.add_argument('--diverse_top_k_by_cluster', action='store_true',
                        help='Selecciona top-K distribuido por cluster origen (manifold_restricted + multi-seed). '
                             'Garantiza que cada seed aporte ≈K/n_seeds candidatos.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolver coeficientes de fitness antes de cargar el modelo.
    alpha, beta = 0.0, 0.0
    if args.fitness_mode == 'combo':
        if args.combo_alpha is not None or args.combo_beta is not None:
            alpha = float(args.combo_alpha or 0.0)
            beta = float(args.combo_beta or 0.0)
        elif args.combo_calibration:
            with open(args.combo_calibration) as f:
                cal = json.load(f)
            alpha = float(cal['alpha']); beta = float(cal['beta'])
        else:
            raise SystemExit("--fitness_mode combo requiere --combo_calibration o (--combo_alpha + --combo_beta)")

    # Las semillas JSON restringen la evolucion alrededor de las geometrías provistas.
    if args.multi_seed_from_jsons:
        args.manifold_restricted = True

    print("=" * 80)
    print(" GA - Diseno inverso")
    print("=" * 80)
    print(f"  pop_size:        {args.pop_size}")
    print(f"  n_gen:           {args.n_gen}")
    print(f"  tournament_k:    {args.tournament_k}")
    print(f"  crossover_rate:  {args.crossover_rate}  (SBX η={args.sbx_eta})")
    print(f"  mutation_rate:   {args.mutation_rate}  (σ={args.mutation_sigma})")
    print(f"  elite_size:      {args.elite_size}")
    print(f"  device:          {device}")
    print(f"  fitness_mode:    {args.fitness_mode}" + (f"  α={alpha:+.4f}  β={beta:+.4f}" if args.fitness_mode == 'combo' else ""))

    # Cargar checkpoint del surrogate.
    ckpt = torch.load(os.path.join(args.model_dir, 'model.pt'), map_location=device, weights_only=False)
    meta = ckpt['meta']
    hidden_dims = ckpt.get('hidden_dims', (512, 512, 512))
    L = meta['L']
    config = ckpt.get('config', {})
    dropout = config.get('dropout', 0.1)
    arch = ckpt.get('arch', 'mlp')
    if arch == 'cnn':
        cnn_channels = ckpt.get('cnn_channels') or (32, 64, 128)
        model = FieldCNN(in_dim=102, out_dim=2 * L,
                         cnn_channels=cnn_channels, fc_hidden=hidden_dims,
                         dropout=dropout).to(device)
    else:
        model = FieldMLP(in_dim=102, out_dim=2 * L, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  arquitectura cargada: {arch}")
    x_scaler = meta['x_scaler']
    print(f"  surrogate cargado: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print()

    # Optional geometry seeds.
    seeds_array = None
    seed_names = None
    if args.multi_seed_from_jsons:
        seed_files = sorted(glob.glob(os.path.join(args.multi_seed_from_jsons, '*.json')))
        seed_list = []
        seed_info = []
        n_filtered_r2 = 0
        for p in seed_files:
            with open(p) as f:
                d = json.load(f)
            widths = np.asarray(d['widths'], dtype=np.float64)
            c2c_raw = d['c2c_d']
            c2c = float(np.mean(c2c_raw)) if isinstance(c2c_raw, list) else float(c2c_raw)
            if len(widths) != 101:
                print(f"  [skip] {os.path.basename(p)}: widths n={len(widths)} != 101")
                continue
            r2_real = float(d.get('r_squared', np.nan))
            if args.seed_min_r2 > 0 and (np.isnan(r2_real) or r2_real < args.seed_min_r2):
                n_filtered_r2 += 1
                continue
            seed_list.append(np.concatenate([widths, [c2c]]))
            seed_info.append((d.get('sim_id') or d.get('cand_id') or os.path.basename(p), r2_real))
        if not seed_list:
            raise SystemExit(f"No se cargaron seeds desde {args.multi_seed_from_jsons} "
                             f"(R²≥{args.seed_min_r2} filtró {n_filtered_r2} JSONs)")
        seeds_array = np.array(seed_list)
        seed_names = [sid for sid, _ in seed_info]
        print(f"  seeds desde JSONs ({args.multi_seed_from_jsons}): {len(seeds_array)} geometrías"
              + (f" (filtrados {n_filtered_r2} con R²<{args.seed_min_r2})" if args.seed_min_r2 > 0 else ""))
        for sid, r2v in seed_info:
            print(f"    {sid:<12}  R²_real={r2v:.4f}")
        print(f"  seed_sigma={args.seed_sigma} µm,  max_deviation={args.max_deviation} µm")
    elif args.seed_from_dataset:
        train = np.load(os.path.join(args.dataset_dir, 'dataset_train.npz'))
        r2 = train['r_squared']
        top_idx = np.argsort(-r2)[:args.seed_top_k]
        seeds_array = train['X'][top_idx]
        print(f"  seeds del train: top-{args.seed_top_k} por R²_paper, mejor R²={r2[top_idx[0]]:.4f}, peor R²={r2[top_idx[-1]]:.4f}")
        print(f"  seed_fraction={args.seed_fraction} (≈{int(args.pop_size*args.seed_fraction)} individuos seeded)")
        print(f"  seed_sigma={args.seed_sigma} µm")

    # Initialize population.
    pop, origin = initialize_population(args.pop_size, rng, seeds=seeds_array,
                                         seed_fraction=args.seed_fraction, seed_sigma=args.seed_sigma,
                                         manifold_restricted=args.manifold_restricted)
    print(f"  población inicial: {len(pop)} individuos válidos")
    if args.manifold_restricted:
        print(f"  MODO RESTRICTED: max_deviation={args.max_deviation} µm desde seed origen")
        print(f"    Distribución de origen: {np.bincount(origin)}")

    # Evolution loop.
    history = []
    t0 = time.time()
    for gen in range(1, args.n_gen + 1):
        fitness, fwhm, conc, r2_pred = compute_fitness_batch(
            pop, model, x_scaler, L, device,
            fitness_mode=args.fitness_mode, alpha=alpha, beta=beta,
        )
        best_idx = int(np.argmax(fitness))
        best_fit = float(fitness[best_idx])
        history.append({
            'gen': gen, 'best_fitness': best_fit,
            'best_r2_pred': float(r2_pred[best_idx]),
            'mean_fitness': float(fitness.mean()),
            'best_fwhm': float(fwhm[best_idx]),
            'best_conc': float(conc[best_idx]),
        })

        if gen % 10 == 0 or gen == 1:
            tag = 'fitness' if args.fitness_mode == 'combo' else 'R²_pred'
            print(f"  Gen {gen:3d}: best {tag}={best_fit:.4f}  R²_pred={r2_pred[best_idx]:.4f}  "
                  f"fwhm={fwhm[best_idx]:.1f}  conc={conc[best_idx]:.2f}  mean {tag}={fitness.mean():.4f}")

        # Seleccion y reproduccion.
        new_pop = []
        new_origin = [] if args.manifold_restricted else None

        if args.manifold_restricted:
            # El modo restringido preserva la cuenca de cada semilla.
            elite_idx = np.argsort(-fitness)[:args.elite_size]
            for i in elite_idx:
                new_pop.append(pop[i].copy())
                new_origin.append(int(origin[i]))
            while len(new_pop) < args.pop_size:
                seed_idx = rng.integers(len(seeds_array))
                cluster_mask = (origin == seed_idx)
                if not cluster_mask.any():
                    # Reconstruir un cluster vacio desde su semilla.
                    base = seeds_array[seed_idx].copy()
                    base[:101] += rng.normal(0, args.seed_sigma, size=101)
                    base[101] += rng.normal(0, args.seed_sigma * 0.5)
                    child = clip_to_physical(base)
                else:
                    cluster_idx = np.where(cluster_mask)[0]
                    chosen = rng.choice(cluster_idx, size=min(args.tournament_k, len(cluster_idx)), replace=False)
                    winner = chosen[np.argmax(fitness[chosen])]
                    child = pop[winner].copy()
                    child = gaussian_mutation(child, args.mutation_rate, args.mutation_sigma, rng)
                child = project_to_manifold(child, seeds_array[seed_idx], args.max_deviation)
                child = clip_to_physical(child)
                new_pop.append(child)
                new_origin.append(int(seed_idx))
            pop = np.array(new_pop)
            origin = np.array(new_origin)
        else:
            # El modo estandar usa cruza global.
            elite_idx = np.argsort(-fitness)[:args.elite_size]
            for i in elite_idx:
                new_pop.append(pop[i].copy())
            while len(new_pop) < args.pop_size:
                p1 = tournament_select(pop, fitness, args.tournament_k, rng)
                p2 = tournament_select(pop, fitness, args.tournament_k, rng)
                if rng.random() < args.crossover_rate:
                    c1, c2 = sbx_crossover(p1, p2, args.sbx_eta, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = gaussian_mutation(c1, args.mutation_rate, args.mutation_sigma, rng)
                c2 = gaussian_mutation(c2, args.mutation_rate, args.mutation_sigma, rng)
                new_pop.append(clip_to_physical(c1))
                if len(new_pop) < args.pop_size:
                    new_pop.append(clip_to_physical(c2))
            pop = np.array(new_pop)

    # Final ranking.
    fitness, fwhm, conc, r2_pred = compute_fitness_batch(
        pop, model, x_scaler, L, device,
        fitness_mode=args.fitness_mode, alpha=alpha, beta=beta,
    )

    if args.diverse_top_k_by_cluster and origin is not None and seeds_array is not None:
        # Balancear candidatos entre cuencas semilla.
        n_seeds_used = len(seeds_array)
        k_per_seed = max(1, (args.top_k + n_seeds_used - 1) // n_seeds_used)
        top_idx_per_cluster = []
        for s in range(n_seeds_used):
            cluster_idx = np.where(origin == s)[0]
            if len(cluster_idx) == 0:
                continue
            ranked = cluster_idx[np.argsort(-fitness[cluster_idx])]
            top_idx_per_cluster.append(ranked[:k_per_seed])
        max_len = max(len(c) for c in top_idx_per_cluster) if top_idx_per_cluster else 0
        interleaved = []
        for j in range(max_len):
            for c in top_idx_per_cluster:
                if j < len(c):
                    interleaved.append(c[j])
        # Completar espacios restantes con los mejores candidatos globales.
        if len(interleaved) < args.top_k:
            remaining_global = [i for i in np.argsort(-fitness) if i not in interleaved]
            interleaved.extend(remaining_global[:args.top_k - len(interleaved)])
        top_idx = np.array(interleaved[:args.top_k])
        print(f"\n  Top-K diversificado por cluster: {k_per_seed} por seed × {n_seeds_used} seeds = {len(top_idx)}")
    else:
        top_idx = np.argsort(-fitness)[:args.top_k]

    print(f"\n  GA completado en {(time.time()-t0)/60:.1f} min")
    tag = 'fitness' if args.fitness_mode == 'combo' else 'R²_pred'
    print(f"  Mejor {tag} final: {fitness[top_idx[0]]:.4f}  R²_pred={r2_pred[top_idx[0]]:.4f}  "
          f"fwhm={fwhm[top_idx[0]]:.1f}, conc={conc[top_idx[0]]:.2f}")
    print()

    # Guardar candidatos en formato de confirmacion FDTD.
    top_csv = os.path.join(args.output_dir, 'top_candidates.csv')
    with open(top_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        header = ['cand_id', 'widths', 'c2c_d', 'r2_pred', 'fwhm_pred', 'concentration_pred', 'fitness']
        if seed_names is not None and origin is not None:
            header.append('seed_origin')
        writer.writerow(header)
        for rank, idx in enumerate(top_idx, start=1):
            ind = pop[idx]
            widths_str = ','.join(f'{w:.6f}' for w in ind[:101])
            row = [f'mvp_c{rank:02d}', widths_str, f'{ind[101]:.6f}',
                   f'{r2_pred[idx]:.4f}', f'{fwhm[idx]:.2f}', f'{conc[idx]:.2f}', f'{fitness[idx]:.4f}']
            if seed_names is not None and origin is not None:
                row.append(seed_names[int(origin[idx])])
            writer.writerow(row)
    print(f"Top-{args.top_k} guardados en {top_csv}")

    # Guardar historial de convergencia.
    conv_path = os.path.join(args.output_dir, 'ga_convergence.json')
    with open(conv_path, 'w') as f:
        json.dump({'history': history, 'config': vars(args)}, f, indent=2)
    print(f"Historia GA en {conv_path}")


if __name__ == '__main__':
    main()
