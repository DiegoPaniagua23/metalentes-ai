"""
Generar la segunda cola del diseno de experimentos.

Genera hasta 500 geometrías de metalente con perfiles de widths variados
(parabólico, gaussiano, lineal, constante, sinusoidal, escalonado) evitando
duplicados respecto al lote 0.1. Todas las geometrías satisfacen las
restricciones físicas: pared mínima de Au ≥ 20 nm y fill factor ∈ [0.1, 0.5].

Uso
---
python src/data_generation/generate_doe_0.2.py [--num_simulations 500] [--seed 2026]
"""

import csv
import argparse
import numpy as np
import os


def load_existing_fingerprints(csv_path):
    """
    Carga fingerprints de geometrías ya generadas para detección de duplicados.

    El fingerprint (c2c_d, w_min, w_max, w_mean) captura la forma global del
    perfil sin almacenar los 101 valores individuales. Colisiones entre perfiles
    genuinamente distintos son infrecuentes en el espacio continuo del DoE.

Parametros
----------
    csv_path : str
        Ruta al CSV existente (separador ;, columnas sim_id;widths;c2c_d).
        Si no existe, retorna un set vacío sin error.

Devuelve
--------
    fingerprints : set of tuple
        Conjunto de tuplas (c2c_d_val, w_min, w_max, w_mean) redondeadas
        a 3-4 decimales para tolerancia de punto flotante.
    """
    fingerprints = set()
    if not os.path.exists(csv_path):
        return fingerprints

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            widths = [float(x) for x in row['widths'].split(',')]
            c2c_vals = [float(x) for x in row['c2c_d'].split(',')]
            fp = (
                round(c2c_vals[0], 3),
                round(min(widths), 3),
                round(max(widths), 3),
                round(np.mean(widths), 4),
            )
            fingerprints.add(fp)
    return fingerprints


def generate_doe(num_simulations=500, num_slits=101, seed=2026,
                         existing_csv=None):
    """
    Genera el CSV de geometrías para el lote del servidor (sim_501 a sim_N).

    Itera hasta max_attempts = 5 * num_simulations veces. En cada intento muestrea
    c2c_d, selecciona un perfil aleatorio, aplica restricciones físicas y verifica
    unicidad por fingerprint. Los perfiles sinusoidal y step son adicionales al DoE
    original 0.1 para ampliar la cobertura del espacio de diseño.

Parametros
----------
    num_simulations : int, optional
        Número de geometrías a generar. Default: 500.
    num_slits : int, optional
        Número de rendijas por metalente. Default: 101.
    seed : int, optional
        Semilla aleatoria. Diferente a la del lote 0.1 para evitar solapamiento.
        Default: 2026.
    existing_csv : str or None, optional
        Ruta al CSV del lote 0.1 para cargar fingerprints y excluir duplicados.
        Default: None.

Devuelve
--------
    None
        Escribe data/doe/simulations_queue_0.2.csv con IDs sim_501 a sim_{500+N}.
        Imprime resumen de geometrías generadas, rechazadas e intentos totales.

Notas
-----
    Los IDs comienzan en sim_501 asumiendo que el lote 0.1 genero sim_001 a sim_500.
    clip() y round(3) se aplican después de generar cada perfil para garantizar
    cumplimiento estricto de los límites físicos (w_min_eff, w_max_eff).
    """
    np.random.seed(seed)

    output_file = "data/doe/simulations_queue_0.2.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_fps = set()
    if existing_csv:
        existing_fps = load_existing_fingerprints(existing_csv)
        print(f"Cargadas {len(existing_fps)} configuraciones existentes para evitar duplicados.")

    # Physical limits shared across DoE queues.
    W_MIN    = 0.020   # 20 nm
    W_MAX    = 0.300   # 300 nm
    C_MIN    = 0.150   # 150 nm
    C_MAX    = 0.300   # 300 nm
    MIN_WALL = 0.020   # pared mínima de Au entre rendijas adyacentes
    F_MIN    = 0.1     # fill factor mínimo (widths / c2c_d)
    F_MAX    = 0.5     # fill factor máximo

    profile_types = ['parabolic', 'gaussian', 'linear', 'constant',
                     'sinusoidal', 'step']

    x = np.linspace(-1, 1, num_slits)
    generated = 0
    attempts = 0
    # Intentos extra compensan geometrias rechazadas.
    max_attempts = num_simulations * 5

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['sim_id', 'widths', 'c2c_d'])

        while generated < num_simulations and attempts < max_attempts:
            attempts += 1

            c2c_d_val = np.round(np.random.uniform(C_MIN, C_MAX), 3)
            c2c_d = [c2c_d_val] * (num_slits - 1)

            # Limites efectivos de ancho para el c2c_d muestreado.
            w_max_eff = min(W_MAX, c2c_d_val - MIN_WALL, c2c_d_val * F_MAX)
            w_min_eff = max(W_MIN, c2c_d_val * F_MIN)

            if w_min_eff >= w_max_eff:
                w_min_eff = W_MIN
                w_max_eff = min(W_MAX, c2c_d_val - MIN_WALL)

            if w_min_eff >= w_max_eff:
                continue

            profile_type = np.random.choice(profile_types)

            if profile_type == 'parabolic':
                a = np.random.uniform(0, 1)
                widths = w_min_eff + (w_max_eff - w_min_eff) * (a * x**2)
            elif profile_type == 'gaussian':
                sigma = np.random.uniform(0.3, 0.8)
                widths = w_min_eff + (w_max_eff - w_min_eff) * np.exp(-x**2 / (2 * sigma**2))
            elif profile_type == 'linear':
                slope = np.random.uniform(0, 1)
                widths = w_min_eff + (w_max_eff - w_min_eff) * np.abs(x * slope)
            elif profile_type == 'sinusoidal':
                freq = np.random.uniform(1, 4)
                phase = np.random.uniform(0, 2 * np.pi)
                envelope = 0.5 * (1 + np.cos(2 * np.pi * freq * x + phase))
                widths = w_min_eff + (w_max_eff - w_min_eff) * envelope
            elif profile_type == 'step':
                # Escalones suaves reducen violaciones locales de pared.
                n_steps = np.random.randint(2, 6)
                step_vals = np.random.uniform(w_min_eff, w_max_eff, n_steps)
                indices = np.linspace(0, num_slits, n_steps + 1, dtype=int)
                widths = np.zeros(num_slits)
                for s in range(n_steps):
                    widths[indices[s]:indices[s+1]] = step_vals[s]
                kernel_size = 5
                kernel = np.ones(kernel_size) / kernel_size
                widths = np.convolve(widths, kernel, mode='same')
            else:  # constant
                val = np.random.uniform(w_min_eff, w_max_eff)
                widths = np.full(num_slits, val)

            # Aplicar limites de ancho despues de generar el perfil.
            widths = np.clip(widths, w_min_eff, w_max_eff)
            widths = np.round(widths, 3)

            # Check pairwise wall gap.
            valid = True
            for j in range(num_slits - 1):
                gap = c2c_d_val - 0.5 * (widths[j] + widths[j + 1])
                if gap < MIN_WALL:
                    valid = False
                    break

            if valid:
                fill_factors = widths / c2c_d_val
                if np.any(fill_factors < F_MIN) or np.any(fill_factors > F_MAX):
                    valid = False

            if not valid:
                continue

            fp = (
                c2c_d_val,
                round(float(widths.min()), 3),
                round(float(widths.max()), 3),
                round(float(widths.mean()), 4),
            )
            if fp in existing_fps:
                continue

            existing_fps.add(fp)
            generated += 1
            sim_id = f"sim_{500 + generated:03d}"

            widths_str = ",".join(map(str, widths.tolist()))
            c2c_d_str = ",".join(map(str, c2c_d))

            writer.writerow([sim_id, widths_str, c2c_d_str])

    print(f"Archivo '{output_file}' generado con {generated} simulaciones nuevas.")
    print(f"  IDs: sim_501 a sim_{500 + generated:03d}")
    print(f"  widths : [{W_MIN:.3f}, {W_MAX:.3f}] µm")
    print(f"  c2c_d  : [{C_MIN:.3f}, {C_MAX:.3f}] µm")
    print(f"  Perfiles: {profile_types}")
    print(f"  Intentos totales: {attempts} (rechazados: {attempts - generated})")

    if generated < num_simulations:
        print(f"  ADVERTENCIA: Solo se generaron {generated}/{num_simulations} "
              f"configuraciones válidas y únicas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera DoE complementario para clúster privado.")
    parser.add_argument("--num_simulations", type=int, default=500,
                        help="Número de simulaciones a generar (default: 500)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Semilla aleatoria (default: 2026)")
    parser.add_argument("--existing_csv", type=str,
                        default="data/doe/simulations_queue_0.1.csv",
                        help="CSV existente para evitar duplicados")
    args = parser.parse_args()

    generate_doe(
        num_simulations=args.num_simulations,
        seed=args.seed,
        existing_csv=args.existing_csv,
    )
