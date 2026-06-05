"""
Propagar un campo focal con el metodo de espectro angular.

Dado el campo complejo en un plano transversal E(x0, y), propaga analíticamente a
planos cercanos en eje x usando ASM (Angular Spectrum Method):

    E(x0 + dx, y) = IFFT(FFT(E(x0, y)) * exp(i * kx * dx))

con kx = sqrt(k0^2 - ky^2) (componente longitudinal del vector de onda).

Esto permite reconstruir una ROI 2D longitudinal a partir del plano focal predicho
por el surrogate, sin re-simular FDTD. Luego se aplica la misma lógica de
extract_features.py (argmax_y por columna + ajuste parabólico) para calcular R²_paper.

Notas
-----
- ASM es exacto en propagación libre (sin estructuras dispersoras en el camino).
- Para metalentes plasmónicas, el plano focal está lejos de la metalente (5-10 µm),
  por lo que la propagación entre planos cercanos al foco es libre y ASM aplica.
- Asume polarización TM (E_y); la propagación escalar de la componente E_y es válida.
"""

from typing import Tuple

import numpy as np


def angular_spectrum_propagate(E_complex: np.ndarray, dx_um: float,
                                wavelength_um: float = 0.63,
                                resolution_per_um: float = 30) -> np.ndarray:
    """
    Propagar un campo complejo 1D una distancia dx_um en x usando ASM.

    Parametros
    ----------
    E_complex : ndarray of complex, shape (ny,)
        Campo complejo en el plano de referencia.
    dx_um : float
        Distancia a propagar en µm (puede ser negativa).
    wavelength_um : float, optional
        Longitud de onda en µm (default 0.63).
    resolution_per_um : float, optional
        Resolución espacial en px/µm (default 30, consistente con Meep).

    Devuelve
    --------
    ndarray of complex, shape (ny,)
        Campo complejo en el plano x0+dx.
    """
    ny = len(E_complex)
    dy_um = 1.0 / resolution_per_um

    # Frecuencias espaciales transversales.
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy_um)

    k0 = 2 * np.pi / wavelength_um

    # Filtrar componentes evanescentes para evitar overflow en propagacion hacia atras.
    kx_sq = k0**2 - ky**2
    propagating = kx_sq >= 0
    kx_real = np.sqrt(np.maximum(kx_sq, 0.0))

    propagator = np.where(propagating, np.exp(1j * kx_real * dx_um), 0.0)

    E_fft = np.fft.fft(E_complex)
    E_prop_fft = E_fft * propagator
    E_prop = np.fft.ifft(E_prop_fft)

    return E_prop


def propagate_to_roi(E_focal_complex: np.ndarray,
                      n_planes_pre: int = 30,
                      n_planes_post: int = 30,
                      dx_um_per_plane: float = 0.2,
                      wavelength_um: float = 0.63,
                      resolution_per_um: float = 30) -> np.ndarray:
    """
    Propagar el campo focal a múltiples planos vecinos en eje x.

    Parametros
    ----------
    E_focal_complex : ndarray of complex, shape (ny,)
        Campo complejo en el plano focal.
    n_planes_pre : int, optional
        Número de planos antes del foco.
    n_planes_post : int, optional
        Número de planos después del foco.
    dx_um_per_plane : float, optional
        Separación entre planos en µm.
    wavelength_um : float, optional
        Longitud de onda en µm.
    resolution_per_um : float, optional
        Resolución espacial en px/µm.

    Devuelve
    --------
    ndarray of complex, shape (n_total, ny)
        Campo complejo por plano (pre, focal, post).
    """
    n_total = n_planes_pre + 1 + n_planes_post
    ny = len(E_focal_complex)
    E_roi = np.zeros((n_total, ny), dtype=np.complex128)

    # Colocar el plano focal en la fila central.
    E_roi[n_planes_pre] = E_focal_complex

    # Planos hacia adelante.
    for i in range(1, n_planes_post + 1):
        dx = i * dx_um_per_plane
        E_roi[n_planes_pre + i] = angular_spectrum_propagate(
            E_focal_complex, dx, wavelength_um, resolution_per_um
        )

    # Planos hacia atras.
    for i in range(1, n_planes_pre + 1):
        dx = -i * dx_um_per_plane
        E_roi[n_planes_pre - i] = angular_spectrum_propagate(
            E_focal_complex, dx, wavelength_um, resolution_per_um
        )

    return E_roi


def r_squared_from_roi(intensity_roi: np.ndarray) -> float:
    """
    Calcular R²_paper del ajuste parabólico a la trayectoria de máximos por columna.

    Replica la lógica de extract_features.py:39-50 pero sobre la ROI propagada.

    Parametros
    ----------
    intensity_roi : ndarray of float, shape (nx, ny)
        Matriz con |E(x, y)|².

    Devuelve
    --------
    float
        R² en [-inf, 1].
    """
    nx = intensity_roi.shape[0]
    y_peaks = np.argmax(intensity_roi, axis=1)
    x_vals = np.arange(nx)

    if x_vals.std() == 0 or y_peaks.std() == 0:
        return 0.0

    coeffs = np.polyfit(x_vals, y_peaks, 2)
    poly = np.poly1d(coeffs)
    y_pred = poly(x_vals)

    ss_res = np.sum((y_peaks - y_pred)**2)
    ss_tot = np.sum((y_peaks - y_peaks.mean())**2)
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def r_squared_from_focal_field(E_focal_complex: np.ndarray,
                                   n_planes_pre: int = 30,
                                   n_planes_post: int = 30,
                                   dx_um_per_plane: float = 0.2,
                                   wavelength_um: float = 0.63,
                                   resolution_per_um: float = 30) -> Tuple[float, np.ndarray]:
    """
    Ejecutar pipeline completo: campo focal, propagación a ROI y R²_paper.

    Parametros
    ----------
    E_focal_complex : ndarray of complex, shape (ny,)
        Campo complejo en el plano focal.
    n_planes_pre : int, optional
        Número de planos antes del foco.
    n_planes_post : int, optional
        Número de planos después del foco.
    dx_um_per_plane : float, optional
        Separación entre planos en µm.
    wavelength_um : float, optional
        Longitud de onda en µm.
    resolution_per_um : float, optional
        Resolución espacial en px/µm.

    Devuelve
    --------
    tuple
        (r_squared, intensity_roi) donde intensity_roi tiene shape (nx, ny).
    """
    E_roi = propagate_to_roi(
        E_focal_complex, n_planes_pre, n_planes_post, dx_um_per_plane,
        wavelength_um, resolution_per_um
    )
    intensity_roi = np.abs(E_roi)**2
    return r_squared_from_roi(intensity_roi), intensity_roi


def amp_phase_to_complex(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    Reconstruir campo complejo desde |E| y Φ.

    Parametros
    ----------
    amplitude : ndarray of float, shape (ny,)
        Magnitud del campo.
    phase : ndarray of float, shape (ny,)
        Fase en radianes.

    Devuelve
    --------
    ndarray of complex, shape (ny,)
        Campo complejo reconstruido.
    """
    return amplitude * np.exp(1j * phase)
