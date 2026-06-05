"""
Entrenar el surrogate multi-output de campo.

Surrogate: geometria (102 dims) -> campo complejo focal (amplitud + fase, 2 x 1024 dims).
Arquitectura baseline: MLP simple (configurable). Loss: MSE separada para amplitud y fase,
con pesos relativos configurables. Optimizer: Adam con ReduceLROnPlateau.

Entradas
--------
data/dataset/dataset_{train,val,test}.npz
    X: (N, 102), amplitude: (N, L), phase: (N, L), r_squared: (N,)

Salidas
-------
results/surrogate/model.pt
    Checkpoint pytorch del mejor modelo (por val loss).
results/surrogate/training_log.json
    Metricas por epoch.
results/surrogate/config.json
    Hiperparametros usados.
"""

import argparse
import json
import os
import sys
import time
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: pytorch no instalado. Actualiza el entorno:")
    print("  micromamba env update -f consultoria_env.yml")
    print("o instala directamente:")
    print("  pip install torch")
    sys.exit(1)


class FieldMLP(nn.Module):
    """
    MLP multi-output: 102 -> hidden... -> 2*L + 1 (amplitud + fase + R2_pred).

    Parametros
    ----------
    in_dim : int
        Dimension de geometria (102).
    out_dim : int
        Longitud del vector campo (L * 2).
    hidden_dims : tuple of int
        Tamano de capas ocultas.
    dropout : float
        Probabilidad de dropout entre capas.
    with_r2_head : bool
        Si True, agrega salida escalar adicional para R2_pred.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(512, 512, 512),
                 dropout: float = 0.1, with_r2_head: bool = True):
        super().__init__()
        self.with_r2_head = with_r2_head
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.field_head = nn.Linear(prev, out_dim)
        if with_r2_head:
            self.r2_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())

    def forward(self, x):
        feat = self.backbone(x)
        field = self.field_head(feat)
        if self.with_r2_head:
            r2 = self.r2_head(feat).squeeze(-1)
            return field, r2
        return field, None


class FieldCNN(nn.Module):
    """
    1D-CNN multi-output: widths (101) como secuencia + c2c_d escalar.

    Notas
    -----
    widths(101) -> reshape(1,101) -> Conv1D(32) -> Conv1D(64) -> Conv1D(128) ->
    GlobalAvgPool -> flatten + c2c_d -> Linear(512) -> Linear(512) ->
    {field_head, r2_head}

    Notas
    -----
    El perfil de widths tiene estructura espacial (rendijas adyacentes correlacionadas).
    Conv1D captura patrones locales que un MLP plano ignora.

    Parametros
    ----------
    in_dim : int
        Dimension de geometria (101 widths + 1 c2c_d = 102).
    out_dim : int
        Longitud del vector campo (L * 2).
    cnn_channels : tuple of int
        Canales de capas convolucionales.
    fc_hidden : tuple of int
        Tamano de capas fully connected post-CNN.
    dropout : float
        Probabilidad de dropout.
    with_r2_head : bool
        Si True, agrega salida escalar adicional para R2_pred.
    """

    def __init__(self, in_dim: int = 102, out_dim: int = 2048,
                 cnn_channels=(32, 64, 128), fc_hidden=(512, 512),
                 dropout: float = 0.1, with_r2_head: bool = True):
        super().__init__()
        self.with_r2_head = with_r2_head
        # Rama convolucional sobre la secuencia de anchos.
        cnn_layers = []
        in_c = 1
        for out_c in cnn_channels:
            cnn_layers.append(nn.Conv1d(in_c, out_c, kernel_size=5, padding=2))
            cnn_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
            in_c = out_c
        self.cnn = nn.Sequential(*cnn_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Rama densa despues de agregar c2c_d.
        fc_layers = []
        prev = cnn_channels[-1] + 1  # +1 por c2c_d
        for h in fc_hidden:
            fc_layers.append(nn.Linear(prev, h))
            fc_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            prev = h
        self.fc = nn.Sequential(*fc_layers)
        self.field_head = nn.Linear(prev, out_dim)
        if with_r2_head:
            self.r2_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())

    def forward(self, x):
        widths = x[:, :101].unsqueeze(1)
        c2c = x[:, 101:102]
        cnn_out = self.cnn(widths)
        pooled = self.pool(cnn_out).squeeze(-1)
        feat_in = torch.cat([pooled, c2c], dim=1)
        feat = self.fc(feat_in)
        field = self.field_head(feat)
        if self.with_r2_head:
            r2 = self.r2_head(feat).squeeze(-1)
            return field, r2
        return field, None


def get_sample_weights(r2_values: np.ndarray, weights_by_band: tuple = (1.0, 10.0, 20.0),
                       thresholds: tuple = (0.70, 0.85)) -> np.ndarray:
    """
    Asignar peso por muestra segun R2_real.

    Parametros
    ----------
    r2_values : ndarray of float, shape (N,)
        R2_real por simulacion.
    weights_by_band : tuple of float, optional
        Pesos para (bajo, medio, alto).
    thresholds : tuple of float, optional
        Umbrales (t_mid, t_high).

    Devuelve
    --------
    ndarray of float, shape (N,)
        Pesos por muestra.

    Notas
    -----
    Por defecto:
    - R2 < 0.70: peso 1.0 (masa central)
    - 0.70 <= R2 < 0.85: peso 10.0 (zona alta)
    - R2 >= 0.85: peso 20.0 (cola superior)
    """
    w_low, w_mid, w_high = weights_by_band
    t_mid, t_high = thresholds
    weights = np.where(r2_values >= t_high, w_high,
                       np.where(r2_values >= t_mid, w_mid, w_low))
    return weights.astype(np.float32)


def fit_input_scaler(X_train: np.ndarray) -> dict:
    """
    Ajustar un min-max scaler basado en train.

    Parametros
    ----------
    X_train : ndarray of float, shape (N, 102)
        Matriz de entrada de train.

    Devuelve
    --------
    dict
        Escalador serializable con 'min' y 'max'.
    """
    return {'min': X_train.min(axis=0).tolist(), 'max': X_train.max(axis=0).tolist()}


def apply_input_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    """
    Aplicar un min-max scaler preajustado.

    Parametros
    ----------
    X : ndarray of float, shape (N, 102)
        Matriz de entrada.
    scaler : dict
        Escalador con 'min' y 'max'.

    Devuelve
    --------
    ndarray of float, shape (N, 102)
        Entrada escalada a [0, 1].
    """
    mn = np.asarray(scaler['min']); mx = np.asarray(scaler['max'])
    rng = np.maximum(mx - mn, 1e-9)
    return (X - mn) / rng


def fit_phase_scaler(phase_train: np.ndarray) -> dict:
    """
    Ajustar escalador para fase usando el std global.

    Parametros
    ----------
    phase_train : ndarray of float, shape (N, L)
        Fase centrada por simulacion.

    Devuelve
    --------
    dict
        Escalador con 'std'.
    """
    std = float(phase_train.std() + 1e-9)
    return {'std': std}


def apply_phase_scaler(phase: np.ndarray, scaler: dict) -> np.ndarray:
    """
    Aplicar escalado por std a la fase.

    Parametros
    ----------
    phase : ndarray of float, shape (N, L)
        Fase centrada.
    scaler : dict
        Escalador con 'std'.

    Devuelve
    --------
    ndarray of float, shape (N, L)
        Fase normalizada.
    """
    return phase / scaler['std']


def make_loaders(args, splits: dict) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Construir DataLoaders y escaladores ajustados en train.

    Parametros
    ----------
    args : argparse.Namespace
        Configuracion de entrenamiento.
    splits : dict
        Reservado para compatibilidad (no usado).

    Devuelve
    --------
    tuple
        (loader_tr, loader_va, loader_te, meta).
    """
    train = np.load(os.path.join(args.dataset_dir, 'dataset_train.npz'))
    val = np.load(os.path.join(args.dataset_dir, 'dataset_val.npz'))
    test = np.load(os.path.join(args.dataset_dir, 'dataset_test.npz'))

    L = int(train['target_length'])

    # Ajustar escaladores solo en train.
    x_scaler = fit_input_scaler(train['X'])
    phase_scaler = fit_phase_scaler(train['phase'])

    def prep(d, use_graded_weights):
        X_n = apply_input_scaler(d['X'], x_scaler).astype(np.float32)
        amp = d['amplitude'].astype(np.float32)
        phase_n = apply_phase_scaler(d['phase'], phase_scaler).astype(np.float32)
        Y = np.concatenate([amp, phase_n], axis=1)
        r2 = d['r_squared'].astype(np.float32)
        # Preferir pesos explicitos del dataset cuando existan.
        if 'sample_weight' in d.files:
            w = d['sample_weight'].astype(np.float32)
        elif use_graded_weights:
            w = get_sample_weights(r2)
        else:
            w = np.ones_like(r2, dtype=np.float32)
        return (torch.from_numpy(X_n), torch.from_numpy(Y),
                torch.from_numpy(r2), torch.from_numpy(w))

    X_tr, Y_tr, R2_tr, W_tr = prep(train, args.graded_weights)
    X_va, Y_va, R2_va, W_va = prep(val, args.graded_weights)
    X_te, Y_te, R2_te, W_te = prep(test, args.graded_weights)

    loader_tr = DataLoader(TensorDataset(X_tr, Y_tr, R2_tr, W_tr), batch_size=args.batch_size, shuffle=True)
    loader_va = DataLoader(TensorDataset(X_va, Y_va, R2_va, W_va), batch_size=args.batch_size, shuffle=False)
    loader_te = DataLoader(TensorDataset(X_te, Y_te, R2_te, W_te), batch_size=args.batch_size, shuffle=False)

    meta = {
        'L': L,
        'x_scaler': x_scaler,
        'phase_scaler': phase_scaler,
        'n_train': len(X_tr), 'n_val': len(X_va), 'n_test': len(X_te),
    }
    return loader_tr, loader_va, loader_te, meta


def compute_loss(field_pred, r2_pred, target_field, target_r2, weights, L, w_amp, w_phase, w_r2):
    """
    Calcular MSE ponderada para amplitud, fase y R2.

    Parametros
    ----------
    field_pred : torch.Tensor, shape (B, 2*L)
        Salida del modelo para campo.
    r2_pred : torch.Tensor or None, shape (B,)
        Salida escalar de R2_pred.
    target_field : torch.Tensor, shape (B, 2*L)
        Campo objetivo (amplitud + fase).
    target_r2 : torch.Tensor, shape (B,)
        R2_real objetivo.
    weights : torch.Tensor, shape (B,)
        Peso por muestra.
    L : int
        Longitud del vector de campo.
    w_amp, w_phase, w_r2 : float
        Pesos de cada termino de loss.

    Devuelve
    --------
    tuple
        (total, loss_amp, loss_phase, loss_r2).
    """
    amp_pred, phase_pred = field_pred[:, :L], field_pred[:, L:]
    amp_true, phase_true = target_field[:, :L], target_field[:, L:]
    # MSE ponderado por muestra.
    mse_amp_per_sample = ((amp_pred - amp_true)**2).mean(dim=1)
    mse_phase_per_sample = ((phase_pred - phase_true)**2).mean(dim=1)
    w_sum = weights.sum() + 1e-12
    loss_amp = (mse_amp_per_sample * weights).sum() / w_sum
    loss_phase = (mse_phase_per_sample * weights).sum() / w_sum
    if r2_pred is not None:
        mse_r2_per_sample = (r2_pred - target_r2)**2
        loss_r2 = (mse_r2_per_sample * weights).sum() / w_sum
    else:
        loss_r2 = torch.tensor(0.0)
    total = w_amp * loss_amp + w_phase * loss_phase + w_r2 * loss_r2
    return total, loss_amp.item(), loss_phase.item(), loss_r2.item()


def run_epoch(model, loader, optimizer, L, w_amp, w_phase, w_r2, device, train: bool):
    """
    Ejecutar una epoca de entrenamiento o evaluacion.

    Parametros
    ----------
    model : torch.nn.Module
        Modelo del surrogate.
    loader : DataLoader
        Loader de datos.
    optimizer : torch.optim.Optimizer
        Optimizador.
    L : int
        Longitud del vector de campo.
    w_amp, w_phase, w_r2 : float
        Pesos de loss.
    device : torch.device
        Dispositivo de ejecucion.
    train : bool
        True para entrenamiento, False para evaluacion.

    Devuelve
    --------
    tuple
        (loss_total, loss_amp, loss_phase, loss_r2) promediados por batch.
    """
    if train:
        model.train()
    else:
        model.eval()
    total, total_amp, total_phase, total_r2, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb, r2b, wb in loader:
            xb, yb, r2b, wb = xb.to(device), yb.to(device), r2b.to(device), wb.to(device)
            field_pred, r2_pred = model(xb)
            loss, la, lp, lr = compute_loss(field_pred, r2_pred, yb, r2b, wb, L, w_amp, w_phase, w_r2)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            total_amp += la
            total_phase += lp
            total_r2 += lr
            n_batches += 1
    return total / n_batches, total_amp / n_batches, total_phase / n_batches, total_r2 / n_batches


def main():
    """
    Entrenar el surrogate multi-output.

    Carga splits estratificados, ajusta escaladores en train, instancia MLP o 1D-CNN,
    entrena con Adam + ReduceLROnPlateau + early stopping, guarda el mejor checkpoint
    por val loss y reporta métricas en test.

    Devuelve
    --------
    None
        Escribe en --output_dir:
        - model.pt          checkpoint del mejor modelo (state_dict + meta).
        - training_log.json historial por epoch + métricas test.
        - config.json       hiperparámetros usados.
    """
    parser = argparse.ArgumentParser(description="Entrena surrogate de campo focal complejo.")
    parser.add_argument('--dataset_dir', default='data/dataset')
    parser.add_argument('--output_dir', default='results/surrogate')
    parser.add_argument('--arch', default='mlp', choices=['mlp', 'cnn'])
    parser.add_argument('--hidden_dims', default='512,512,512', help='Tamanos ocultos separados por coma (backbone MLP o FC de CNN)')
    parser.add_argument('--cnn_channels', default='32,64,128', help='Canales Conv1D (solo si --arch cnn)')
    parser.add_argument('--graded_weights', action='store_true',
                        help='Activa pesos graduados por R²_real (x1/x10/x20 para R²<0.7/<0.85/>=0.85)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_amp', type=float, default=1.0)
    parser.add_argument('--weight_phase', type=float, default=1.0)
    parser.add_argument('--weight_r2', type=float, default=10.0,
                        help='Peso del MSE de R²_pred (loss auxiliar). Alto para priorizar R² preciso.')
    parser.add_argument('--patience', type=int, default=30, help='Paciencia para early stopping')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(" ENTRENAMIENTO SURROGATE DE CAMPO FOCAL")
    print("=" * 80)
    print(f"  device:        {device}")
    print(f"  hidden_dims:   {args.hidden_dims}")
    print(f"  dropout:       {args.dropout}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  lr:            {args.lr}")
    print(f"  weight_amp/ph/r2: {args.weight_amp} / {args.weight_phase} / {args.weight_r2}")
    print(f"  epochs:        {args.epochs}")
    print(f"  early stop:    patience={args.patience}")
    print()

    # Cargar splits y escaladores.
    splits = {}
    loader_tr, loader_va, loader_te, meta = make_loaders(args, splits)
    L = meta['L']
    print(f"  Dataset: N_train={meta['n_train']}, N_val={meta['n_val']}, N_test={meta['n_test']}, L={L}")
    print()

    # Construir modelo.
    hidden_dims = tuple(int(h) for h in args.hidden_dims.split(','))
    if args.arch == 'mlp':
        model = FieldMLP(in_dim=102, out_dim=2 * L, hidden_dims=hidden_dims,
                         dropout=args.dropout, with_r2_head=True).to(device)
        arch_desc = f"MLP {hidden_dims}"
    elif args.arch == 'cnn':
        cnn_channels = tuple(int(c) for c in args.cnn_channels.split(','))
        model = FieldCNN(in_dim=102, out_dim=2 * L,
                         cnn_channels=cnn_channels, fc_hidden=hidden_dims,
                         dropout=args.dropout, with_r2_head=True).to(device)
        arch_desc = f"1D-CNN canales={cnn_channels} fc={hidden_dims}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Modelo: {arch_desc}, params totales = {n_params:,}")
    print(f"  Pesos graduados por R²: {args.graded_weights}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Ciclo de entrenamiento.
    best_val = float('inf')
    best_epoch = -1
    patience_counter = 0
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_amp, tr_phase, tr_r2 = run_epoch(model, loader_tr, optimizer, L,
                                                       args.weight_amp, args.weight_phase, args.weight_r2,
                                                       device, train=True)
        va_loss, va_amp, va_phase, va_r2 = run_epoch(model, loader_va, optimizer, L,
                                                       args.weight_amp, args.weight_phase, args.weight_r2,
                                                       device, train=False)
        scheduler.step(va_loss)
        lr_curr = optimizer.param_groups[0]['lr']

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': meta,
                'hidden_dims': hidden_dims,
                'arch': args.arch,
                'cnn_channels': tuple(int(c) for c in args.cnn_channels.split(',')) if args.arch == 'cnn' else None,
                'config': vars(args),
                'epoch': epoch,
                'val_loss': va_loss,
            }, os.path.join(args.output_dir, 'model.pt'))
        else:
            patience_counter += 1

        msg = (f"[{epoch:3d}/{args.epochs}] "
               f"train: {tr_loss:.4f} (a {tr_amp:.4f} p {tr_phase:.4f} r2 {tr_r2:.4f})  "
               f"val: {va_loss:.4f} (a {va_amp:.4f} p {va_phase:.4f} r2 {va_r2:.4f})  "
               f"lr={lr_curr:.1e}")
        if is_best:
            msg += "  best"
        print(msg)

        history.append({
            'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss,
            'train_amp': tr_amp, 'train_phase': tr_phase, 'train_r2': tr_r2,
            'val_amp': va_amp, 'val_phase': va_phase, 'val_r2': va_r2,
            'lr': lr_curr, 'is_best': is_best,
        })

        if patience_counter >= args.patience:
            print(f"\n  Parada temprana (sin mejora en {args.patience} epochs)")
            break

    # Evaluar el mejor checkpoint en test.
    print()
    print(f"  Mejor epoch: {best_epoch} con val_loss = {best_val:.4f}")
    print(f"  Tiempo total: {(time.time() - t0)/60:.1f} min")

    ckpt = torch.load(os.path.join(args.output_dir, 'model.pt'))
    model.load_state_dict(ckpt['model_state_dict'])
    te_loss, te_amp, te_phase, te_r2 = run_epoch(model, loader_te, optimizer, L,
                                                   args.weight_amp, args.weight_phase, args.weight_r2,
                                                   device, train=False)
    print(f"  TEST: loss={te_loss:.4f}, amp={te_amp:.4f}, phase={te_phase:.4f}, r2={te_r2:.4f}")

    # Persistir metadatos de entrenamiento.
    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump({
            'history': history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val,
            'test_loss': te_loss, 'test_amp': te_amp, 'test_phase': te_phase, 'test_r2': te_r2,
            'config': vars(args), 'hidden_dims': hidden_dims, 'n_params': n_params,
            'meta': {'L': L, 'n_train': meta['n_train'], 'n_val': meta['n_val'], 'n_test': meta['n_test']},
        }, f, indent=2)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump({'args': vars(args), 'hidden_dims': hidden_dims}, f, indent=2)
    print(f"\nModelo y metricas guardados en {args.output_dir}/")


if __name__ == '__main__':
    main()
