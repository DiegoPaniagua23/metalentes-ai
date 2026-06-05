"""
generate_figures.py — Genera las figuras del reporte técnico.

Produce las gráficas comparativas IA vs FDTD y de resultados a partir de los
artefactos del proyecto (JSONs confirmados, dataset, candidatos finales).

Ejecutar desde la raíz del proyecto:
    micromamba run -n consultoria_env python report/Figuras/generate_figures.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUT = 'report/Figuras'
os.makedirs(OUT, exist_ok=True)

# Paleta común del reporte: sobria, legible en impresión y consistente entre
# figuras de datos y esquemas TikZ.
PALETTE = {
    'primary': '#1F5A85',      # azul técnico
    'secondary': '#2A9D8F',    # verde/teal para casos favorables
    'accent': '#C26A2E',       # naranja para fase o énfasis secundario
    'risk': '#B6403A',         # rojo para umbrales/riesgo
    'neutral': '#6B7280',      # gris texto/guías
    'light': '#D8DEE9',        # gris claro para distribuciones base
    'dark': '#263238',         # casi negro
    'gold': '#C58A2A',         # oro/Au
}

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 9.5,
    'axes.titlesize': 10.5,
    'axes.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'axes.edgecolor': PALETTE['neutral'],
    'axes.linewidth': 0.8,
    'grid.color': '#C9D1D9',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.45,
})


def style_ax(ax, grid_axis='both'):
    """Aplicar estilo común de ejes a todas las gráficas Matplotlib."""
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['neutral'])
    ax.spines['bottom'].set_color(PALETTE['neutral'])


def savefig(fig, name):
    """Guardar figura con configuración uniforme."""
    fig.savefig(f'{OUT}/{name}.png')
    plt.close(fig)


def load_confirmed(dirs):
    """Cargar pares (r2_pred, r2_real, cand_id) de directorios de confirmados."""
    rows = []
    for d in dirs:
        for fp in sorted(glob.glob(os.path.join(d, '*.json'))):
            dd = json.load(open(fp))
            rp = dd.get('r2_pred_mvp')
            rr = dd.get('r_squared')
            if rp is not None and rr is not None:
                rows.append((dd.get('cand_id', os.path.basename(fp)), float(rp), float(rr)))
    return rows


# ---------------------------------------------------------------------------
# Fig 1 — Comparación IA vs FDTD: R²_pred (surrogate) vs R²_real (FDTD)
# ---------------------------------------------------------------------------
def fig_pred_vs_real():
    rows = load_confirmed(['data/confirmed/iter3', 'data/confirmed/iter4', 'data/confirmed/iter5'])
    if not rows:
        print('  [pred_vs_real] sin datos, omitido')
        return
    rp = np.array([r[1] for r in rows])
    rr = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(rr, rp, s=45, alpha=0.78, edgecolor=PALETTE['dark'], linewidth=0.35,
               color=PALETTE['primary'])
    lim = [0, 1]
    ax.plot(lim, lim, color=PALETTE['dark'], linestyle='--', linewidth=1,
            label='$R^2_{pred} = R^2_{real}$ (ideal)')
    ax.axhline(0.90, color=PALETTE['risk'], linestyle=':', linewidth=1.0, alpha=0.85)
    ax.axvline(0.90, color=PALETTE['risk'], linestyle=':', linewidth=1.0, alpha=0.85,
               label='umbral $R^2=0.90$')
    ax.set_xlabel('$R^2_{real}$ (FDTD)')
    ax.set_ylabel('$R^2_{pred}$ (surrogate)')
    ax.set_title('Predicción del surrogate vs. simulación FDTD')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(loc='lower right', fontsize=9)
    style_ax(ax)
    savefig(fig, 'fig_pred_vs_real')
    print(f'  [pred_vs_real] {len(rows)} candidatos confirmados')


# ---------------------------------------------------------------------------
# Fig 2 — Histograma de R² del dataset
# ---------------------------------------------------------------------------
def fig_hist_r2():
    f = np.load('data/dataset/fields.npz', allow_pickle=True)
    r2 = f['r_squared']
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(r2, bins=50, color=PALETTE['primary'], alpha=0.78,
            edgecolor=PALETTE['dark'], linewidth=0.25)
    ax.axvline(0.90, color=PALETTE['risk'], linestyle='--', linewidth=1.2,
               label='umbral $R^2=0.90$')
    ax.set_xlabel('$R^2$ (ajuste parabólico, FDTD)')
    ax.set_ylabel('Número de simulaciones')
    ax.set_title(f'Distribución de $R^2$ en el dataset (N={len(r2)})')
    ax.legend()
    style_ax(ax, grid_axis='y')
    savefig(fig, 'fig_hist_r2')
    print(f'  [hist_r2] N={len(r2)}, R²>=0.90: {(r2>=0.90).sum()}')


# ---------------------------------------------------------------------------
# Fig — Curvas de entrenamiento del surrogate (pérdida train/val por época)
# ---------------------------------------------------------------------------
def fig_training_curves():
    """Curvas de pérdida del MLP multi-salida a partir del log de entrenamiento.

    Panel (a): pérdida total de entrenamiento vs.\\ validación por época, con la mejor
    época marcada (criterio de early stopping).
    Panel (b): componentes de la pérdida de validación (amplitud, fase, R^2_pred) en
    escala normalizada, evidenciando que la fase domina y converge primero.
    """
    fp = 'results/surrogate/training_log.json'
    if not os.path.exists(fp):
        print('  [training_curves] log no encontrado, omitido')
        return
    d = json.load(open(fp))
    h = d['history']
    ep = [r['epoch'] for r in h]
    best = d['best_epoch']

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    ax.plot(ep, [r['train_loss'] for r in h], color=PALETTE['primary'], lw=1.6, label='entrenamiento')
    ax.plot(ep, [r['val_loss'] for r in h], color=PALETTE['accent'], lw=1.6, label='validación')
    ax.axvline(best, color=PALETTE['neutral'], ls='--', lw=1, label='mejor época (%d)' % best)
    ax.set_xlabel('Época'); ax.set_ylabel('Pérdida total $\\mathcal{L}$')
    ax.set_title('(a) Curvas de pérdida total')
    ax.legend(fontsize=8); style_ax(ax)

    ax = axes[1]
    for key, lab, c in [('val_amp', 'amplitud', PALETTE['primary']),
                        ('val_phase', 'fase', PALETTE['accent']),
                        ('val_r2', '$\\hat{R}^2_{pred}$', PALETTE['secondary'])]:
        ax.plot(ep, [r[key] for r in h], lw=1.4, color=c, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('Época'); ax.set_ylabel('MSE de validación (escala normalizada)')
    ax.set_title('(b) Componentes de la pérdida')
    ax.legend(fontsize=8); style_ax(ax)

    fig.tight_layout()
    savefig(fig, 'fig_training_curves')
    print('  [training_curves] %d épocas, mejor=%d, val_loss=%.3f'
          % (len(ep), best, d['best_val_loss']))


# ---------------------------------------------------------------------------
# Fig EDA — Caracterización del campo óptico focal (amplitud + fase)
# ---------------------------------------------------------------------------
def fig_eda_campos():
    """Caracteriza los objetivos densos del surrogate: amplitud |E|(y) y fase Φ(y).

    Panel (a): perfiles de amplitud alineados al pico — enfocadores vs no enfocadores.
    Panel (b): perfiles de fase desenvuelta y centrada — enfocadores vs no enfocadores.
    Panel (c): distribución del FWHM del lóbulo de amplitud por grupo.
    Panel (d): distribución del rango dinámico de la fase por grupo.
    """
    f = np.load('data/dataset/fields.npz', allow_pickle=True)
    amp, ph, r2 = f['amplitude'], f['phase'], f['r_squared']
    N, L = amp.shape

    def fwhm(a):
        idx = np.where(a >= 0.5 * a.max())[0]
        return (idx.max() - idx.min() + 1) if len(idx) else 0

    fw = np.array([fwhm(amp[i]) for i in range(N)])
    rng_ph = ph.max(1) - ph.min(1)
    foc = r2 >= 0.85          # enfocadores
    nof = r2 < 0.30           # no enfocadores

    # Ejemplos representativos: 3 mejores enfocadores y 3 no enfocadores medianos.
    foc_ex = np.argsort(r2)[-3:][::-1]
    nof_idx = np.where(nof)[0]
    rng = np.random.default_rng(0)
    nof_ex = rng.choice(nof_idx, size=3, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    cf, cn = PALETTE['primary'], PALETTE['light']

    # (a) amplitud promedio por grupo, alineada al pico
    ax = axes[0, 0]
    half = 220
    yy = np.arange(-half, half)

    def aligned_stack(idxs):
        M = np.full((len(idxs), 2 * half), np.nan)
        for k, i in enumerate(idxs):
            pk = np.argmax(amp[i])
            lo, hi = max(0, pk - half), min(L, pk + half)
            M[k, (lo - pk) + half:(hi - pk) + half] = amp[i, lo:hi]
        return np.nanmean(M, 0), np.nanstd(M, 0)

    for mask, c, lab in [(nof, cn, 'no enfocador ($R^2<0.30$)'),
                         (foc, cf, 'enfocador ($R^2\\geq0.85$)')]:
        m, s = aligned_stack(np.where(mask)[0])
        ax.fill_between(yy, np.clip(m - s, 0, None), m + s, color=c, alpha=0.25)
        ax.plot(yy, m, color=c, linewidth=1.6, label=lab)
    ax.set_xlabel('Posición transversal relativa al pico (índice)')
    ax.set_ylabel('$|E|(y)$ promedio (normalizada)')
    ax.set_title('(a) Amplitud focal media $\\pm 1\\sigma$ por grupo')
    style_ax(ax)
    ax.legend(fontsize=8)

    # (b) fase
    ax = axes[0, 1]
    for i in nof_ex:
        ax.plot(ph[i], color=cn, linewidth=1, alpha=0.8)
    for i in foc_ex:
        ax.plot(ph[i], color=cf, linewidth=1.4)
    ax.set_xlabel('Posición transversal $y$ (índice)')
    ax.set_ylabel('$\\Phi(y)$ (rad, centrada)')
    ax.set_title('(b) Fase focal desenvuelta')
    style_ax(ax)

    # (c) FWHM por grupo
    ax = axes[1, 0]
    bins = np.linspace(0, L, 40)
    ax.hist(fw[nof], bins=bins, color=cn, alpha=0.7, density=True, label='no enfocador')
    ax.hist(fw[foc], bins=bins, color=cf, alpha=0.7, density=True, label='enfocador')
    ax.set_xlabel('FWHM del lóbulo de amplitud (índices)')
    ax.set_ylabel('Densidad')
    ax.set_title('(c) Anchura del lóbulo focal')
    ax.legend(fontsize=8); style_ax(ax, grid_axis='y')

    # (d) rango de fase por grupo
    ax = axes[1, 1]
    bins = np.linspace(0, rng_ph.max(), 40)
    ax.hist(rng_ph[nof], bins=bins, color=cn, alpha=0.7, density=True, label='no enfocador')
    ax.hist(rng_ph[foc], bins=bins, color=cf, alpha=0.7, density=True, label='enfocador')
    ax.set_xlabel('Rango dinámico de la fase $\\max\\Phi-\\min\\Phi$ (rad)')
    ax.set_ylabel('Densidad')
    ax.set_title('(d) Excursión de la fase')
    ax.legend(fontsize=8); style_ax(ax, grid_axis='y')

    fig.tight_layout()
    savefig(fig, 'fig_eda_campos')
    print(f'  [eda_campos] FWHM foc/nof={fw[foc].mean():.0f}/{fw[nof].mean():.0f}, '
          f'rango fase foc/nof={rng_ph[foc].mean():.1f}/{rng_ph[nof].mean():.1f}')


# ---------------------------------------------------------------------------
# Fig 3 — Aguja vs amplia: distribución de R² por cuenca (hill-climbing)
# ---------------------------------------------------------------------------
def fig_cuencas():
    basins = {'627 (amplia)': 'data/confirmed/iterhc627',
              '042 (aguja)': 'data/confirmed/iterhc042',
              '721 (aguja)': 'data/confirmed/iterhc721'}
    data, labels = [], []
    for lab, d in basins.items():
        rs = [json.load(open(fp))['r_squared'] for fp in glob.glob(os.path.join(d, '*.json'))]
        if rs:
            data.append(rs); labels.append(f'{lab}\n(n={len(rs)})')
    if not data:
        print('  [cuencas] sin datos, omitido')
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    colors = [PALETTE['secondary'], PALETTE['risk'], PALETTE['risk']]
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(0.90, color=PALETTE['risk'], linestyle='--', linewidth=1,
               label='umbral $R^2=0.90$')
    ax.axhline(0.70, color=PALETTE['neutral'], linestyle=':', linewidth=0.9,
               label='colapso de enfoque ($R^2<0.70$)')
    ax.set_ylabel('$R^2_{real}$ (FDTD)')
    ax.set_title('Robustez por cuenca: perturbaciones $\\sigma=5$ nm')
    ax.legend(fontsize=9, loc='lower center')
    style_ax(ax, grid_axis='y')
    savefig(fig, 'fig_cuencas_robustez')
    print(f'  [cuencas] {len(data)} cuencas')


# ---------------------------------------------------------------------------
# Fig 4 y 5 — Geometría y campo focal de hc627_036 (recomendada)
# ---------------------------------------------------------------------------
def fig_geometria_campo():
    fp = 'data/confirmed/iterhc627/hc627_036_features.json'
    if not os.path.exists(fp):
        print('  [hc627_036] no encontrado, omitido')
        return
    d = json.load(open(fp))
    widths = np.array(d['widths']) * 1000  # nm
    trans = np.array(d['transmittance'])
    r2val = float(d['r_squared'])
    titulo_perfil = 'Perfil de anchos — hc627_036 ($R^2=%.4f$)' % r2val

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(1, len(widths) + 1), widths, '-o', ms=3,
            color=PALETTE['primary'], markerfacecolor='white', markeredgewidth=0.8,
            linewidth=1.2)
    ax.set_xlabel('Índice de rendija')
    ax.set_ylabel('Ancho (nm)')
    ax.set_title(titulo_perfil)
    style_ax(ax)
    savefig(fig, 'fig_perfil_hc627')

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(trans, color=PALETTE['accent'], linewidth=1.2)
    ax.set_xlabel('Posición transversal $y$ (índice)')
    ax.set_ylabel('Intensidad normalizada $|E|^2$')
    ax.set_title('Perfil de intensidad en el plano focal — hc627_036')
    style_ax(ax)
    savefig(fig, 'fig_campo_hc627')
    print(f'  [hc627_036] perfil + campo (R²={d["r_squared"]:.4f})')


# ---------------------------------------------------------------------------
# Fig — Proyección PCA del espacio de diseño coloreada por R²
# ---------------------------------------------------------------------------
def fig_pca_espacio():
    """Proyección PCA 2D de las geometrías (102-D) coloreada por R²_real.

    Evidencia empírica de que las geometrías de alta calidad se concentran en pocas
    regiones separadas (las cuencas), motivando el manifold y las 3 cuencas.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    f = np.load('data/dataset/fields.npz', allow_pickle=True)
    X, r2, ids = f['X'], f['r_squared'], f['sim_ids']
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(6.6, 5))
    order = np.argsort(r2)                       # alto R² encima
    sc = ax.scatter(Z[order, 0], Z[order, 1], c=r2[order], s=14, cmap='viridis',
                    alpha=0.86, edgecolor='none')
    cb = fig.colorbar(sc, ax=ax); cb.set_label('$R^2_{real}$ (FDTD)')
    id2i = {str(s): i for i, s in enumerate(ids)}
    for sid, lab in [('sim_042', '042'), ('sim_721', '721'), ('sim_627', '627')]:
        if sid in id2i:
            i = id2i[sid]
            ax.scatter(Z[i, 0], Z[i, 1], s=150, marker='*', color=PALETTE['risk'],
                       edgecolor=PALETTE['dark'], linewidth=0.6, zorder=5)
            ax.annotate('cuenca %s' % lab, (Z[i, 0], Z[i, 1]), textcoords='offset points',
                        xytext=(7, 5), fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Componente principal 1 (%.0f%% var.)' % ev[0])
    ax.set_ylabel('Componente principal 2 (%.0f%% var.)' % ev[1])
    ax.set_title('Espacio de diseño (PCA) coloreado por $R^2_{real}$')
    style_ax(ax)
    savefig(fig, 'fig_pca_espacio')
    print('  [pca] var. explicada 2D = %.0f%% (%.0f+%.0f)' % (ev[:2].sum(), ev[0], ev[1]))


# ---------------------------------------------------------------------------
# Fig — Conceptual: cuenca "aguja" vs "meseta" bajo tolerancia de fabricación
# ---------------------------------------------------------------------------
def fig_aguja_meseta():
    """Ilustración conceptual del hallazgo central, anclada a los valores reales.

    Cuenca aguja (042): pico alto (0.958) pero colapsa bajo ±5 nm (media 0.36).
    Cuenca meseta (627): pico 0.926, robusta a ±5 nm (media 0.73).
    """
    x = np.linspace(-15, 15, 500)               # perturbación en nm
    needle = 0.32 + (0.958 - 0.32) * np.exp(-(x**2) / (2 * 2.3**2))
    plateau = 0.60 + (0.926 - 0.60) * np.exp(-(x**2) / (2 * 9.5**2))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(-5, 5, color=PALETTE['neutral'], alpha=0.15, label='tolerancia $\\pm5$ nm')
    ax.plot(x, needle, color=PALETTE['risk'], lw=2.2, label='cuenca aguja (042/721)')
    ax.plot(x, plateau, color=PALETTE['secondary'], lw=2.2, label='cuenca meseta (627)')
    ax.axhline(0.90, color=PALETTE['risk'], ls='--', lw=0.9, alpha=0.75)
    ax.text(14.5, 0.905, '$R^2=0.90$', color=PALETTE['risk'], fontsize=8, ha='right', va='bottom')
    ax.axhline(0.70, color=PALETTE['neutral'], ls=':', lw=0.9)
    ax.text(14.5, 0.705, 'colapso ($R^2<0.70$)', color=PALETTE['neutral'], fontsize=8, ha='right', va='bottom')
    # anotaciones de los picos reales
    ax.annotate('máx. 0.958', (0, 0.958), textcoords='offset points', xytext=(8, -2),
                fontsize=8, color=PALETTE['risk'])
    ax.annotate('máx. 0.926', (0, 0.926), textcoords='offset points', xytext=(-58, 6),
                fontsize=8, color=PALETTE['secondary'])
    ax.set_xlabel('Perturbación de la geometría respecto al óptimo (nm)')
    ax.set_ylabel('$R^2_{real}$')
    ax.set_title('Cuenca «aguja» vs «meseta»: efecto de la tolerancia de fabricación')
    ax.set_ylim(0.2, 1.0); ax.set_xlim(-15, 15)
    ax.legend(loc='lower center', fontsize=8.5); style_ax(ax)
    savefig(fig, 'fig_aguja_meseta')
    print('  [aguja_meseta] figura conceptual generada')


# ---------------------------------------------------------------------------
# Fig — Overlay predicción del surrogate vs campo real FDTD
# ---------------------------------------------------------------------------
def fig_surrogate_overlay():
    """Superpone |E|(y) y Φ(y) predichos por el surrogate sobre el campo real FDTD."""
    import sys
    sys.path.insert(0, 'src/surrogate')
    try:
        import torch
        from train_field_surrogate import FieldMLP, FieldCNN, apply_input_scaler
    except Exception as e:
        print('  [overlay] torch/modelo no disponible (%s), omitido' % e)
        return
    ckpt = torch.load('results/surrogate/model.pt', map_location='cpu', weights_only=False)
    meta = ckpt['meta']; L = meta['L']; xs = meta['x_scaler']; ps = meta['phase_scaler']
    hidden = ckpt.get('hidden_dims', (512, 512, 512)); arch = ckpt.get('arch', 'mlp')
    dropout = ckpt.get('config', {}).get('dropout', 0.1)
    if arch == 'cnn':
        model = FieldCNN(in_dim=102, out_dim=2 * L, fc_hidden=hidden, dropout=dropout)
    else:
        model = FieldMLP(in_dim=102, out_dim=2 * L, hidden_dims=hidden, dropout=dropout)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()

    d = np.load('data/dataset/dataset_test.npz', allow_pickle=True)
    X, amp_real, ph_real, r2, ids = d['X'], d['amplitude'], d['phase'], d['r_squared'], d['sim_ids']
    # Predecir todo el test para elegir un enfocador REPRESENTATIVO (no la aguja atípica):
    # entre los buenos enfocadores (R²>=0.6), el de error de amplitud mediano.
    Xn_all = apply_input_scaler(X, xs).astype(np.float32)
    with torch.no_grad():
        fp_all, _ = model(torch.from_numpy(Xn_all))
    fp_all = fp_all.numpy()
    amp_all = fp_all[:, :L]
    mse_amp = ((amp_all - amp_real) ** 2).mean(axis=1)
    cand = np.where(r2 >= 0.6)[0]
    i = int(cand[np.argsort(mse_amp[cand])[len(cand) // 2]])  # mediana de error en zona alta
    ph_pred = fp_all[i, L:] * ps['std']
    amp_pred = amp_all[i]
    y = np.arange(L)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(y, amp_real[i], color=PALETTE['dark'], lw=1.3, label='FDTD (real)')
    axes[0].plot(y, amp_pred, color=PALETTE['primary'], lw=1.3, ls='--', label='surrogate')
    axes[0].set_xlabel('Posición transversal $y$ (índice)')
    axes[0].set_ylabel('$|E|(y)$ (normalizada)')
    axes[0].set_title('(a) Amplitud'); axes[0].legend(fontsize=8); style_ax(axes[0])
    axes[1].plot(y, ph_real[i], color=PALETTE['dark'], lw=1.3, label='FDTD (real)')
    axes[1].plot(y, ph_pred, color=PALETTE['accent'], lw=1.3, ls='--', label='surrogate')
    axes[1].set_xlabel('Posición transversal $y$ (índice)')
    axes[1].set_ylabel('$\\Phi(y)$ (rad, centrada)')
    axes[1].set_title('(b) Fase'); axes[1].legend(fontsize=8); style_ax(axes[1])
    fig.suptitle('Predicción del surrogate vs. FDTD — %s ($R^2_{real}=%.3f$, MSE$_{|E|}$=%.3f)'
                 % (str(ids[i]), float(r2[i]), float(mse_amp[i])), fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig_surrogate_overlay')
    print('  [overlay] %s (R²=%.3f, MSE_amp=%.4f)' % (str(ids[i]), float(r2[i]), float(mse_amp[i])))


if __name__ == '__main__':
    print('Generando figuras del reporte...')
    fig_pred_vs_real()
    fig_hist_r2()
    fig_eda_campos()
    fig_training_curves()
    fig_cuencas()
    fig_geometria_campo()
    fig_pca_espacio()
    fig_aguja_meseta()
    fig_surrogate_overlay()
    print(f'Figuras guardadas en {OUT}/')
