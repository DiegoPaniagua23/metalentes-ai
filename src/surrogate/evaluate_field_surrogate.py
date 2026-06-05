"""
Evaluar el surrogate de campo en los splits de validacion y prueba.

Calcula:
- MSE de amplitud y fase en val y test.
- FWHM (Full Width at Half Maximum) del perfil de |E|^2 predicho vs real.
- Correlacion entre FWHM predicho y R2_real (proxy util).
- Concentracion del foco (max/mean) predicho vs real.

Notas
-----
El surrogate predice el campo en el plano focal. Las metricas auxiliares se
calculan desde ese plano y, opcionalmente, mediante propagacion angular spectrum.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: pytorch no instalado. Actualiza el entorno (consultoria_env.yml).")
    sys.exit(1)

# Imports locales para ejecutar los scripts sin instalar el paquete.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_processing'))
from train_field_surrogate import FieldMLP, FieldCNN, apply_input_scaler, apply_phase_scaler
from propagation import r_squared_from_focal_field

warnings.filterwarnings('ignore', message='.*overflow.*')


def fwhm_transverse(intensity: np.ndarray) -> float:
    """
    Calcular Full-Width-at-Half-Maximum del perfil 1D de intensidad.

    Parametros
    ----------
    intensity : ndarray of float, shape (L,)
        Intensidad transversal |E|^2.

    Devuelve
    --------
    float
        Ancho en indices (entre puntos donde I = max/2).

    Notas
    -----
    Si el perfil no decae a max/2, retorna len(intensity).
    """
    if intensity.max() <= 0:
        return float(len(intensity))
    half = intensity.max() / 2
    above = intensity >= half
    if not above.any():
        return float(len(intensity))
    idx = np.where(above)[0]
    return float(idx[-1] - idx[0] + 1)


def focal_concentration(intensity: np.ndarray) -> float:
    """
    Calcular concentracion del foco: max/mean del perfil de intensidad.

    Parametros
    ----------
    intensity : ndarray of float, shape (L,)
        Intensidad transversal.

    Devuelve
    --------
    float
        Concentracion (max/mean).
    """
    m = intensity.mean()
    return float(intensity.max() / (m + 1e-12))


def parabolic_r2_proxy(intensity: np.ndarray) -> float:
    """
    Proxy de R2_paper: ajuste parabolico al perfil de intensidad transversal.

    Parametros
    ----------
    intensity : ndarray of float, shape (L,)
        Intensidad transversal |E|^2.

    Devuelve
    --------
    float
        R2 del ajuste parabolico en ventana alrededor del pico.

    Notas
    -----
    No es el R2_paper estricto (requiere trayectoria longitudinal de maximos),
    pero es discriminativo en la zona del pico.
    """
    n = len(intensity)
    # Ajustar solo una ventana local alrededor del pico.
    peak_idx = int(np.argmax(intensity))
    W = max(20, n // 20)
    a = max(0, peak_idx - W)
    b = min(n, peak_idx + W + 1)
    y = intensity[a:b]
    x = np.arange(len(y))
    if len(y) < 5 or y.std() == 0:
        return 0.0
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1.0 - ss_res / (ss_tot + 1e-9)


def evaluate_split(model, split_path: str, x_scaler, phase_scaler, L, device, name: str):
    """
    Evaluar el surrogate sobre un split y calcular metricas.

    Parametros
    ----------
    model : torch.nn.Module
        Surrogate multi-output.
    split_path : str
        Ruta a dataset_{val,test}.npz.
    x_scaler : dict
        Escalador de entradas.
    phase_scaler : dict
        Escalador de fase.
    L : int
        Longitud del vector de campo.
    device : torch.device
        Dispositivo de ejecucion.
    name : str
        Etiqueta del split (VAL/TEST).

    Devuelve
    --------
    dict
        Metricas agregadas y por simulacion.
    """
    data = np.load(split_path, allow_pickle=True)
    sim_ids = data['sim_ids']
    X = data['X']
    amp_real = data['amplitude']
    phase_real_centered = data['phase']
    r2_real = data['r_squared']
    # La resolucion efectiva depende de la longitud transversal original.
    ny_original = data['ny_original'] if 'ny_original' in data.files else np.full(len(sim_ids), L)

    X_n = apply_input_scaler(X, x_scaler).astype(np.float32)
    phase_real_norm = apply_phase_scaler(phase_real_centered, phase_scaler).astype(np.float32)

    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_n).to(device)
        field_pred, r2_pred_tensor = model(x_tensor)
        field_pred = field_pred.cpu().numpy()
        r2_pred_direct = r2_pred_tensor.cpu().numpy() if r2_pred_tensor is not None else np.full(len(X_n), np.nan)
    amp_pred = field_pred[:, :L]
    phase_pred_norm = field_pred[:, L:]
    # Restaurar la escala de fase usada durante el entrenamiento.
    phase_pred = phase_pred_norm * phase_scaler['std']

    N = len(sim_ids)

    # Errores de campo por simulacion.
    mse_amp = np.mean((amp_pred - amp_real)**2, axis=1)
    mse_phase = np.mean((phase_pred - phase_real_centered)**2, axis=1)

    # Intensidades predichas y de referencia.
    int_pred = np.maximum(amp_pred, 0)**2
    int_real = amp_real**2

    # Metricas auxiliares del plano focal.
    fwhm_p = np.array([fwhm_transverse(i) for i in int_pred])
    fwhm_r = np.array([fwhm_transverse(i) for i in int_real])
    conc_p = np.array([focal_concentration(i) for i in int_pred])
    conc_r = np.array([focal_concentration(i) for i in int_real])
    r2par_p = np.array([parabolic_r2_proxy(i) for i in int_pred])
    r2par_r = np.array([parabolic_r2_proxy(i) for i in int_real])

    # La propagacion usa la resolucion efectiva posterior al remuestreo.
    n_pre, n_post = 35, 35
    dx_um = 0.2
    E_pred_complex = amp_pred * np.exp(1j * phase_pred)
    E_real_complex = amp_real * np.exp(1j * phase_real_centered)
    r2asm_p = np.zeros(N)
    r2asm_r = np.zeros(N)
    for i in range(N):
        ny_orig = int(ny_original[i])
        resolution_eff = L * 30.0 / ny_orig
        r2asm_p[i], _ = r_squared_from_focal_field(
            E_pred_complex[i].astype(np.complex128),
            n_planes_pre=n_pre, n_planes_post=n_post,
            dx_um_per_plane=dx_um, wavelength_um=0.63, resolution_per_um=resolution_eff
        )
        r2asm_r[i], _ = r_squared_from_focal_field(
            E_real_complex[i].astype(np.complex128),
            n_planes_pre=n_pre, n_planes_post=n_post,
            dx_um_per_plane=dx_um, wavelength_um=0.63, resolution_per_um=resolution_eff
        )

    print(f"\n=== {name}  (N={N}) ===")
    print(f"  MSE amplitud:    mean={mse_amp.mean():.4f}  median={np.median(mse_amp):.4f}  max={mse_amp.max():.4f}")
    print(f"  MSE fase:        mean={mse_phase.mean():.4f}  median={np.median(mse_phase):.4f}  max={mse_phase.max():.4f}")
    print(f"  FWHM real:       mean={fwhm_r.mean():.1f}  median={np.median(fwhm_r):.1f}  min={fwhm_r.min():.1f}")
    print(f"  FWHM pred:       mean={fwhm_p.mean():.1f}  median={np.median(fwhm_p):.1f}  min={fwhm_p.min():.1f}")
    print(f"  Concentración real: mean={conc_r.mean():.1f}  max={conc_r.max():.1f}")
    print(f"  Concentración pred: mean={conc_p.mean():.1f}  max={conc_p.max():.1f}")

    # Correlaciones entre metricas derivadas y R2 FDTD.
    def safe_corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9: return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    corr_fwhm_p_r2 = safe_corr(-fwhm_p, r2_real)
    corr_fwhm_r_r2 = safe_corr(-fwhm_r, r2_real)
    corr_conc_p_r2 = safe_corr(conc_p, r2_real)
    corr_conc_r_r2 = safe_corr(conc_r, r2_real)
    corr_r2par_p_r2 = safe_corr(r2par_p, r2_real)
    corr_r2par_r_r2 = safe_corr(r2par_r, r2_real)
    # R2 propagado desde campos predichos y de referencia.
    corr_r2asm_p_r2 = safe_corr(r2asm_p, r2_real)
    corr_r2asm_r_r2 = safe_corr(r2asm_r, r2_real)
    # Error absoluto medio del R2 propagado.
    mae_r2asm_p = float(np.mean(np.abs(r2asm_p - r2_real)))
    mae_r2asm_r = float(np.mean(np.abs(r2asm_r - r2_real)))

    print(f"\n  CORRELACIONES con R²_real:")
    print(f"    -FWHM(pred)      -> R²_real:  {corr_fwhm_p_r2:+.4f}  (proxy transversal)")
    print(f"    -FWHM(real)      -> R²_real:  {corr_fwhm_r_r2:+.4f}  (techo transversal)")
    print(f"    concentr.(pred)  -> R²_real:  {corr_conc_p_r2:+.4f}")
    print(f"    concentr.(real)  -> R²_real:  {corr_conc_r_r2:+.4f}  (techo)")
    print(f"    R²_parab(pred)   -> R²_real:  {corr_r2par_p_r2:+.4f}")
    print(f"    R²_parab(real)   -> R²_real:  {corr_r2par_r_r2:+.4f}  (techo)")
    print(f"\n  R²_ASM propagado (longitudinal):")
    print(f"    R²_ASM(pred)     -> R²_real:  {corr_r2asm_p_r2:+.4f}    MAE = {mae_r2asm_p:.4f}")
    print(f"    R²_ASM(real)     -> R²_real:  {corr_r2asm_r_r2:+.4f}    MAE = {mae_r2asm_r:.4f}  (techo)")

    # Salida escalar directa usada por el fitness del GA.
    corr_r2direct_r2 = safe_corr(r2_pred_direct, r2_real)
    mae_r2direct = float(np.mean(np.abs(r2_pred_direct - r2_real)))
    rmse_r2direct = float(np.sqrt(np.mean((r2_pred_direct - r2_real)**2)))
    print(f"\n  R²_pred DIRECTO (output escalar del surrogate, FITNESS GA):")
    print(f"    R²_pred_directo  -> R²_real:  {corr_r2direct_r2:+.4f}    MAE = {mae_r2direct:.4f}    RMSE = {rmse_r2direct:.4f}")
    print(f"    range R²_pred:   [{r2_pred_direct.min():.4f}, {r2_pred_direct.max():.4f}]   range R²_real: [{r2_real.min():.4f}, {r2_real.max():.4f}]")
    # Diagnostico en el subconjunto de R2 alto.
    mask_top = r2_real >= 0.7
    if mask_top.sum() > 3:
        print(f"\n  En zona R²_real ≥ 0.7 (n={mask_top.sum()}):")
        print(f"    corr(R²_pred_directo, R²_real):  {safe_corr(r2_pred_direct[mask_top], r2_real[mask_top]):+.4f}")
        print(f"    mean |R²_pred_directo - R²_real|: {float(np.mean(np.abs(r2_pred_direct[mask_top] - r2_real[mask_top]))):.4f}")
        print(f"    corr(R²_ASM(pred), R²_real):  {safe_corr(r2asm_p[mask_top], r2_real[mask_top]):+.4f}")
        print(f"    mean |R²_ASM(pred) - R²_real|: {float(np.mean(np.abs(r2asm_p[mask_top] - r2_real[mask_top]))):.4f}")

    return {
        'name': name, 'N': int(N),
        'mse_amp_mean': float(mse_amp.mean()),
        'mse_amp_median': float(np.median(mse_amp)),
        'mse_phase_mean': float(mse_phase.mean()),
        'mse_phase_median': float(np.median(mse_phase)),
        'corr_fwhm_pred_r2': corr_fwhm_p_r2,
        'corr_fwhm_real_r2': corr_fwhm_r_r2,
        'corr_conc_pred_r2': corr_conc_p_r2,
        'corr_conc_real_r2': corr_conc_r_r2,
        'corr_r2par_pred_r2': corr_r2par_p_r2,
        'corr_r2par_real_r2': corr_r2par_r_r2,
        'corr_r2asm_pred_r2': corr_r2asm_p_r2,
        'corr_r2asm_real_r2': corr_r2asm_r_r2,
        'mae_r2asm_pred': mae_r2asm_p,
        'mae_r2asm_real': mae_r2asm_r,
        'corr_r2direct_r2': corr_r2direct_r2,
        'mae_r2direct': mae_r2direct,
        'rmse_r2direct': rmse_r2direct,
        'per_sim': {
            'sim_ids': sim_ids.tolist(),
            'r2_real': r2_real.tolist(),
            'r2asm_pred': r2asm_p.tolist(),
            'r2asm_real': r2asm_r.tolist(),
            'fwhm_pred': fwhm_p.tolist(),
            'mse_amp': mse_amp.tolist(),
            'mse_phase': mse_phase.tolist(),
        }
    }


def main():
    """
    Cargar el checkpoint del surrogate y evaluar val y test.

    Reconstruye la arquitectura desde el meta del checkpoint (MLP o CNN), corre la
    inferencia sobre ambos splits, calcula MSE de campo, métricas auxiliares
    (FWHM, concentración, R²_parab proxy), R²_ASM por propagación FFT y la
    correlación R²_pred ↔ R²_real (la métrica clave para validar el fitness GA).

    Devuelve
    --------
    None
        Escribe results/surrogate/eval_metrics.json con todas las métricas.
        Imprime al stdout las correlaciones por split y la zona R²≥0.7.
    """
    parser = argparse.ArgumentParser(description="Evalúa surrogate en val y test.")
    parser.add_argument('--model_dir', default='results/surrogate')
    parser.add_argument('--dataset_dir', default='data/dataset')
    parser.add_argument('--output_dir', default='results/surrogate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print(" EVALUACIÓN SURROGATE")
    print("=" * 80)

    # Cargar checkpoint y reconstruir la arquitectura guardada.
    ckpt_path = os.path.join(args.model_dir, 'model.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no existe {ckpt_path}. Corre train_field_surrogate.py primero.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt['meta']
    hidden_dims = ckpt.get('hidden_dims', (512, 512, 512))
    L = meta['L']

    # Replicar exactamente la arquitectura usada en entrenamiento.
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
    print(f"  Arquitectura cargada: {arch}")

    print(f"  Modelo cargado: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"  Arquitectura: MLP {hidden_dims}, L={L}")

    results = {}
    for name, fname in [('VAL', 'dataset_val.npz'), ('TEST', 'dataset_test.npz')]:
        results[name] = evaluate_split(
            model, os.path.join(args.dataset_dir, fname),
            meta['x_scaler'], meta['phase_scaler'], L, device, name
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'eval_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetricas guardadas en {out_path}")


if __name__ == '__main__':
    main()
