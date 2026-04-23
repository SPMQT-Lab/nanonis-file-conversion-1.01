"""
ProbeFlow — image processing pipeline for STM/SXM data.

All functions operate on raw float32/float64 2-D arrays (physical units).
They are intentionally free of any GUI dependency so they can be called from
worker threads or batch scripts without importing PySide6.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    label as _nd_label,
    laplace as _nd_laplace,
)

# ── Font path for scale-bar labels ────────────────────────────────────────────
_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  remove_bad_lines
# ═════════════════════════════════════════════════════════════════════════════

def remove_bad_lines(arr: np.ndarray, threshold_mad: float = 5.0) -> np.ndarray:
    """
    Replace outlier scan lines via weighted interpolation from neighbours.

    A row is "bad" when |row_median − overall_median| > threshold_mad × MAD,
    where MAD is the median absolute deviation of per-row medians.
    Bad rows are replaced by a distance-weighted blend of the nearest good
    rows above and below (falls back to the single nearest good row when
    only one side is available).
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    row_meds = np.array([
        float(np.nanmedian(arr[r])) for r in range(Ny)
    ])

    finite_meds = row_meds[np.isfinite(row_meds)]
    if finite_meds.size == 0:
        return arr

    overall_med = float(np.median(finite_meds))
    mad = float(np.median(np.abs(finite_meds - overall_med)))
    if mad == 0.0:
        return arr

    bad_mask = np.abs(row_meds - overall_med) > threshold_mad * mad
    bad_rows = np.where(bad_mask)[0]

    if bad_rows.size == 0:
        return arr

    good_rows = np.where(~bad_mask)[0]
    if good_rows.size == 0:
        return arr

    for r in bad_rows:
        above = good_rows[good_rows < r]
        below = good_rows[good_rows > r]

        if above.size > 0 and below.size > 0:
            ra, rb = int(above[-1]), int(below[0])
            da, db = r - ra, rb - r
            wa = db / (da + db)
            wb = da / (da + db)
            arr[r] = wa * arr[ra] + wb * arr[rb]
        elif above.size > 0:
            arr[r] = arr[int(above[-1])]
        else:
            arr[r] = arr[int(below[0])]

    return arr


# ═════════════════════════════════════════════════════════════════════════════
# 2.  subtract_background
# ═════════════════════════════════════════════════════════════════════════════

def subtract_background(arr: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Fit and subtract a polynomial background.

    order=1  → plane    (ax + by + c)
    order=2  → full 2nd-degree (ax² + by² + cxy + dx + ey + f)

    Coordinates are normalised to [-1, 1] for numerical stability.
    Only finite pixels participate in the least-squares fit.
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    ys = np.linspace(-1.0, 1.0, Ny)
    xs = np.linspace(-1.0, 1.0, Nx)
    Xg, Yg = np.meshgrid(xs, ys)

    flat_x = Xg.ravel()
    flat_y = Yg.ravel()
    flat_z = arr.ravel()

    finite = np.isfinite(flat_z)
    if finite.sum() < (3 if order == 1 else 6):
        return arr

    if order == 1:
        A = np.column_stack([flat_x[finite], flat_y[finite],
                             np.ones(finite.sum())])
    else:
        fx, fy = flat_x[finite], flat_y[finite]
        A = np.column_stack([fx**2, fy**2, fx*fy, fx, fy,
                             np.ones(finite.sum())])

    b = flat_z[finite]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if order == 1:
        bg = coeffs[0]*Xg + coeffs[1]*Yg + coeffs[2]
    else:
        bg = (coeffs[0]*Xg**2 + coeffs[1]*Yg**2 +
              coeffs[2]*Xg*Yg  + coeffs[3]*Xg    +
              coeffs[4]*Yg     + coeffs[5])

    return arr - bg


# ═════════════════════════════════════════════════════════════════════════════
# 3.  align_rows  (Gwyddion: "Align Rows")
# ═════════════════════════════════════════════════════════════════════════════

def align_rows(arr: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Fix inter-line DC offsets by subtracting a per-row reference value.

    method='median'  — subtract each row's median (robust to tip crashes)
    method='mean'    — subtract each row's mean
    method='linear'  — fit and subtract a per-row linear trend (slope + offset)

    This is the most effective first step for raw STM data where each scan
    line has an independent height offset due to thermal drift or tip jumps.
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    if method == 'median':
        for r in range(Ny):
            row = arr[r]
            ref = float(np.nanmedian(row))
            if np.isfinite(ref):
                arr[r] -= ref

    elif method == 'mean':
        for r in range(Ny):
            row = arr[r]
            ref = float(np.nanmean(row))
            if np.isfinite(ref):
                arr[r] -= ref

    elif method == 'linear':
        xs = np.linspace(-1.0, 1.0, Nx)
        for r in range(Ny):
            row = arr[r]
            fin = np.isfinite(row)
            if fin.sum() < 2:
                continue
            coeffs = np.polyfit(xs[fin], row[fin], 1)
            arr[r] -= np.polyval(coeffs, xs)

    return arr


# ═════════════════════════════════════════════════════════════════════════════
# 4.  facet_level  (Gwyddion: "Facet Level")
# ═════════════════════════════════════════════════════════════════════════════

def facet_level(arr: np.ndarray, threshold_deg: float = 3.0) -> np.ndarray:
    """
    Level the image using only the nearly-flat (horizontal) pixels as
    the reference plane.

    Local slopes are estimated via finite differences.  Pixels with a slope
    angle below *threshold_deg* are treated as part of flat terraces and used
    for the plane fit.  The fitted plane is then subtracted from the whole image.
    This avoids step edges biasing the background correction — essential for
    Au(111), Si(111) and other stepped surfaces.
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    if Ny < 3 or Nx < 3:
        return arr

    # Estimate local gradient via central differences (pixel units)
    gy, gx = np.gradient(arr)

    # Convert threshold from degrees to tangent value
    tan_thresh = math.tan(math.radians(threshold_deg))
    slope_mag = np.sqrt(gx**2 + gy**2)
    flat_mask = (slope_mag < tan_thresh) & np.isfinite(arr)

    if flat_mask.sum() < 3:
        # Not enough flat pixels — fall back to full-image plane
        return subtract_background(arr, order=1)

    ys = np.linspace(-1.0, 1.0, Ny)
    xs = np.linspace(-1.0, 1.0, Nx)
    Xg, Yg = np.meshgrid(xs, ys)

    flat_x = Xg[flat_mask]
    flat_y = Yg[flat_mask]
    flat_z = arr[flat_mask]

    A = np.column_stack([flat_x, flat_y, np.ones(flat_x.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, flat_z, rcond=None)
    bg = coeffs[0]*Xg + coeffs[1]*Yg + coeffs[2]

    return arr - bg


# ═════════════════════════════════════════════════════════════════════════════
# 5.  fourier_filter
# ═════════════════════════════════════════════════════════════════════════════

def fourier_filter(
    arr:    np.ndarray,
    mode:   str   = 'low_pass',
    cutoff: float = 0.1,
    window: str   = 'hanning',
) -> np.ndarray:
    """
    Apply a 2-D FFT filter.

    cutoff  — fraction of Nyquist [0, 1].
    mode    — 'low_pass' | 'high_pass'
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    mean_val = float(np.nanmean(arr))
    arr[~np.isfinite(arr)] = mean_val

    if window == 'hanning':
        wy = np.hanning(Ny)
        wx = np.hanning(Nx)
    elif window == 'hamming':
        wy = np.hamming(Ny)
        wx = np.hamming(Nx)
    else:
        wy = np.ones(Ny)
        wx = np.ones(Nx)

    win2d = np.outer(wy, wx)
    windowed = arr * win2d

    F = np.fft.fft2(windowed)
    F = np.fft.fftshift(F)

    cy, cx = Ny / 2.0, Nx / 2.0
    yr = (np.arange(Ny) - cy) / cy
    xr = (np.arange(Nx) - cx) / cx
    Xr, Yr = np.meshgrid(xr, yr)
    R = np.sqrt(Xr**2 + Yr**2)

    if mode == 'low_pass':
        mask = (R <= cutoff).astype(np.float64)
    else:
        mask = (R >= cutoff).astype(np.float64)

    F_filtered = F * mask
    F_filtered = np.fft.ifftshift(F_filtered)
    result = np.fft.ifft2(F_filtered).real

    safe_win = np.where(win2d > 1e-6, win2d, 1.0)
    result = result / safe_win

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 6.  gaussian_smooth  (Gwyddion: "Gaussian filter")
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_smooth(arr: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """
    Apply a 2-D isotropic Gaussian smoothing filter.

    sigma_px — standard deviation in pixels.  Typical STM values: 0.5–3.
    NaN pixels are handled by weighted normalisation so they don't propagate.
    """
    arr = arr.astype(np.float64, copy=True)

    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        fill = float(np.nanmean(arr))
        arr[nan_mask] = fill

    smoothed = gaussian_filter(arr, sigma=sigma_px, mode='reflect')

    if nan_mask.any():
        smoothed[nan_mask] = np.nan

    return smoothed


# ═════════════════════════════════════════════════════════════════════════════
# 7.  edge_detect  (Gwyddion/Tycoon: Laplacian, DoG, LoG)
# ═════════════════════════════════════════════════════════════════════════════

def edge_detect(
    arr:    np.ndarray,
    method: str   = 'laplacian',
    sigma:  float = 1.0,
    sigma2: float = 2.0,
) -> np.ndarray:
    """
    Edge / feature enhancement using Laplacian-family filters.

    method='laplacian' — discrete Laplacian (2nd derivative, no smoothing)
    method='log'       — Laplacian of Gaussian  (σ = sigma)
    method='dog'       — Difference of Gaussians (σ₁=sigma, σ₂=sigma2)

    Returns the filter response — positive = bright edges/peaks,
    negative = dark edges/valleys.  Useful for atomic-resolution contrast
    enhancement and finding adsorption sites.
    """
    a = arr.astype(np.float64, copy=True)
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        a[nan_mask] = float(np.nanmean(a))

    if method == 'laplacian':
        result = _nd_laplace(a)

    elif method == 'log':
        # Pre-smooth then Laplacian
        smoothed = gaussian_filter(a, sigma=max(sigma, 0.1), mode='reflect')
        result   = _nd_laplace(smoothed)

    elif method == 'dog':
        g1 = gaussian_filter(a, sigma=max(sigma,  0.1), mode='reflect')
        g2 = gaussian_filter(a, sigma=max(sigma2, sigma + 0.1), mode='reflect')
        result = g1 - g2

    else:
        raise ValueError(f"Unknown edge_detect method: {method!r}")

    if nan_mask.any():
        result[nan_mask] = np.nan

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 8.  gmm_autoclip  (UniMR: GMM auto-thresholding)
# ═════════════════════════════════════════════════════════════════════════════

def gmm_autoclip(arr: np.ndarray, n_samples: int = 2000) -> tuple[float, float]:
    """
    Estimate optimal clip_low / clip_high percentiles using a 2-component
    Gaussian Mixture Model fitted to the image histogram.

    Returns (clip_low_pct, clip_high_pct) as percentile values [0–100],
    suitable for passing directly to np.percentile.

    The approach mirrors the UniMR project's gmm_threshold():  fit two
    Gaussians to the value distribution, find their intersection, and map
    the lower-component tail and upper-component tail to clip percentiles.
    Falls back to (1.0, 99.0) if the fit is degenerate.
    """
    data = arr.astype(np.float64).ravel()
    data = data[np.isfinite(data)]
    if data.size < 10:
        return 1.0, 99.0

    # Subsample for speed
    if data.size > n_samples:
        rng = np.random.default_rng(0)
        data = rng.choice(data, size=n_samples, replace=False)

    # EM for a 2-component 1-D GMM (numpy-only implementation)
    data_min, data_max = data.min(), data.max()
    if data_max <= data_min:
        return 1.0, 99.0

    # Initialise: split at median
    med = float(np.median(data))
    mu1, mu2 = float(data[data <= med].mean()), float(data[data > med].mean())
    s1 = s2 = float(data.std()) / 2.0 + 1e-10
    pi1 = pi2 = 0.5

    for _ in range(60):
        # E-step: responsibilities
        def _gauss(x, mu, s):
            return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

        r1 = pi1 * _gauss(data, mu1, s1)
        r2 = pi2 * _gauss(data, mu2, s2)
        denom = r1 + r2 + 1e-300
        r1 /= denom
        r2 /= denom

        # M-step
        n1, n2 = r1.sum(), r2.sum()
        if n1 < 1e-6 or n2 < 1e-6:
            break
        mu1_new = (r1 * data).sum() / n1
        mu2_new = (r2 * data).sum() / n2
        s1_new  = math.sqrt((r1 * (data - mu1_new)**2).sum() / n1) + 1e-10
        s2_new  = math.sqrt((r2 * (data - mu2_new)**2).sum() / n2) + 1e-10
        pi1_new = n1 / (n1 + n2)
        pi2_new = n2 / (n1 + n2)

        if (abs(mu1_new - mu1) < 1e-8 * abs(data_max - data_min) and
                abs(mu2_new - mu2) < 1e-8 * abs(data_max - data_min)):
            mu1, mu2, s1, s2, pi1, pi2 = mu1_new, mu2_new, s1_new, s2_new, pi1_new, pi2_new
            break
        mu1, mu2, s1, s2, pi1, pi2 = mu1_new, mu2_new, s1_new, s2_new, pi1_new, pi2_new

    # Ensure mu1 < mu2
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        s1,  s2  = s2,  s1
        pi1, pi2 = pi2, pi1

    # Convert GMM component extents to percentiles on the full dataset
    full_data = arr.astype(np.float64).ravel()
    full_data = full_data[np.isfinite(full_data)]
    if full_data.size == 0:
        return 1.0, 99.0

    # Low clip: 2σ below lower component mean
    low_val  = mu1 - 2.0 * s1
    # High clip: 2σ above upper component mean
    high_val = mu2 + 2.0 * s2

    clip_low  = float(np.sum(full_data <  low_val)  / full_data.size * 100.0)
    clip_high = float(np.sum(full_data <= high_val) / full_data.size * 100.0)

    # Clamp to sane range
    clip_low  = max(0.0, min(clip_low,  10.0))
    clip_high = min(100.0, max(clip_high, 90.0))

    return clip_low, clip_high


# ═════════════════════════════════════════════════════════════════════════════
# 9.  detect_grains  (Gwyddion: "Mark Grains by Threshold / Watershed")
# ═════════════════════════════════════════════════════════════════════════════

def detect_grains(
    arr:                np.ndarray,
    threshold_pct:      float = 50.0,
    above:              bool  = True,
    min_grain_px:       int   = 5,
) -> tuple[np.ndarray, int, dict]:
    """
    Detect grains/islands by thresholding the height data.

    Parameters
    ----------
    arr             : 2-D float array (height data)
    threshold_pct   : percentile of data used as threshold (0–100)
    above           : True = grains are above threshold (islands on flat terrace)
                      False = grains are below (holes/depressions)
    min_grain_px    : grains smaller than this many pixels are discarded

    Returns
    -------
    label_map  : int32 array with each grain labelled 1, 2, 3, …  (0 = background)
    n_grains   : number of grains found
    stats      : dict with 'areas_px', 'centroids', 'mean_heights'
    """
    a = arr.astype(np.float64, copy=True)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        empty = np.zeros(a.shape, dtype=np.int32)
        return empty, 0, {}

    thresh = float(np.percentile(finite, threshold_pct))

    if above:
        binary = np.isfinite(a) & (a >= thresh)
    else:
        binary = np.isfinite(a) & (a <= thresh)

    label_map, n_raw = _nd_label(binary)

    # Remove grains below minimum size
    if min_grain_px > 1:
        for grain_id in range(1, n_raw + 1):
            mask = label_map == grain_id
            if mask.sum() < min_grain_px:
                label_map[mask] = 0

        # Re-label contiguously
        label_map, n_grains = _nd_label(label_map > 0)
    else:
        n_grains = n_raw

    # Compute per-grain statistics
    areas, centroids, heights = [], [], []
    for grain_id in range(1, n_grains + 1):
        mask = label_map == grain_id
        ys, xs = np.where(mask)
        areas.append(int(mask.sum()))
        centroids.append((float(xs.mean()), float(ys.mean())))
        vals = a[mask]
        heights.append(float(np.nanmean(vals[np.isfinite(vals)])))

    stats = {
        'areas_px':    areas,
        'centroids':   centroids,
        'mean_heights': heights,
    }

    return label_map.astype(np.int32), n_grains, stats


# ═════════════════════════════════════════════════════════════════════════════
# 10.  measure_periodicity
# ═════════════════════════════════════════════════════════════════════════════

def measure_periodicity(
    arr:           np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    n_peaks:       int = 5,
) -> list[dict]:
    """
    Find dominant spatial periodicities in a 2-D array using its power spectrum.

    Returns a list (length ≤ n_peaks) of dicts:
        {'period_m': float, 'angle_deg': float, 'strength': float}
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    mean_val = float(np.nanmean(arr))
    arr[~np.isfinite(arr)] = mean_val

    wy = np.hanning(Ny)
    wx = np.hanning(Nx)
    win2d = np.outer(wy, wx)

    F = np.fft.fft2(arr * win2d)
    F = np.fft.fftshift(F)
    power = np.abs(F) ** 2

    cy, cx = Ny // 2, Nx // 2

    DC_R = 2.0

    half_mask = np.zeros((Ny, Nx), dtype=bool)
    half_mask[:cy, :] = True

    yr = np.arange(Ny) - cy
    xr = np.arange(Nx) - cx
    Xr, Yr = np.meshgrid(xr.astype(float), yr.astype(float))
    R_px = np.sqrt(Xr**2 + Yr**2)
    half_mask[R_px < DC_R] = False

    search_power = power.copy()
    search_power[~half_mask] = 0.0

    results = []
    suppress_r = max(3, min(Ny, Nx) // 20)

    for _ in range(n_peaks):
        idx = int(np.argmax(search_power))
        py, px = divmod(idx, Nx)

        peak_val = float(search_power[py, px])
        if peak_val <= 0:
            break

        fy = (py - cy) / Ny
        fx = (px - cx) / Nx

        f_mag = math.sqrt(fx**2 + fy**2)
        if f_mag == 0.0:
            break

        freq_m_x = fx / pixel_size_x_m
        freq_m_y = fy / pixel_size_y_m
        freq_m   = math.sqrt(freq_m_x**2 + freq_m_y**2)
        period_m = 1.0 / freq_m if freq_m > 0 else 0.0

        angle_deg = math.degrees(math.atan2(fy * pixel_size_y_m,
                                             fx * pixel_size_x_m))

        results.append({
            'period_m':  period_m,
            'angle_deg': angle_deg,
            'strength':  peak_val,
        })

        for (rpy, rpx) in [(py, px), (Ny - py, Nx - px)]:
            for dy in range(-suppress_r, suppress_r + 1):
                for dx in range(-suppress_r, suppress_r + 1):
                    ny_ = int(rpy) + dy
                    nx_ = int(rpx) + dx
                    if 0 <= ny_ < Ny and 0 <= nx_ < Nx:
                        search_power[ny_, nx_] = 0.0

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 11.  export_png
# ═════════════════════════════════════════════════════════════════════════════

_NICE_STEPS_NM = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
    100, 200, 500, 1000, 2000, 5000, 10000,
]


def _pick_scalebar_length(width_m: float, image_px: int,
                          target_frac: float = 0.20,
                          unit: str = 'nm') -> tuple[float, str]:
    unit_factors = {'nm': 1e9, 'Å': 1e10, 'pm': 1e12}
    factor = unit_factors.get(unit, 1e9)

    target_m = width_m * target_frac
    target_u = target_m * factor

    best = _NICE_STEPS_NM[0]
    for s in _NICE_STEPS_NM:
        if abs(s - target_u) < abs(best - target_u):
            best = s
        if s > target_u * 2:
            break

    bar_m = best / factor

    if best == int(best):
        label = f"{int(best)} {unit}"
    else:
        label = f"{best:g} {unit}"

    return bar_m, label


def export_png(
    arr:           np.ndarray,
    out_path,
    colormap_key:  str,
    clip_low:      float,
    clip_high:     float,
    lut_fn,
    scan_range_m:  tuple,
    add_scalebar:  bool  = True,
    scalebar_unit: str   = 'nm',
    scalebar_pos:  str   = 'bottom-right',
) -> None:
    """
    Export a full-resolution colourised image with an optional scale bar.

    lut_fn(colormap_key) must return a (256, 3) uint8 LUT array.
    scan_range_m  — (width_m, height_m); scale bar is skipped when width ≤ 0.
    """
    from PIL import Image as _Image, ImageDraw as _IDraw, ImageFont as _IFont

    arr = arr.astype(np.float64, copy=True)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Array contains no finite values.")

    vmin = float(np.percentile(finite, clip_low))
    vmax = float(np.percentile(finite, clip_high))
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    safe = np.where(np.isfinite(arr), arr, vmin).astype(np.float64)
    u8   = np.clip((safe - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    lut  = lut_fn(colormap_key)
    colored = lut[u8]
    img = _Image.fromarray(colored, mode="RGB")

    width_m = scan_range_m[0] if len(scan_range_m) >= 1 else 0.0
    Ny, Nx  = arr.shape

    if add_scalebar and width_m > 0:
        bar_m, bar_label = _pick_scalebar_length(
            width_m, Nx, target_frac=0.20, unit=scalebar_unit)

        bar_px = int(round(bar_m / width_m * Nx))
        bar_px = max(4, min(bar_px, Nx - 20))

        font_size = max(12, Ny // 40)
        font = None
        if _FONT_PATH.exists():
            try:
                font = _IFont.truetype(str(_FONT_PATH), size=font_size)
            except Exception:
                pass
        if font is None:
            font = _IFont.load_default()

        MARGIN      = 10
        BAR_HEIGHT  = max(4, Ny // 80)
        TEXT_GAP    = 3

        dummy_img  = _Image.new("RGB", (1, 1))
        dummy_draw = _IDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), bar_label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if scalebar_pos == 'bottom-left':
            bar_x0 = MARGIN
        else:
            bar_x0 = Nx - MARGIN - bar_px

        bar_y0 = Ny - MARGIN - BAR_HEIGHT
        bar_x1 = bar_x0 + bar_px
        bar_y1 = bar_y0 + BAR_HEIGHT

        text_x = bar_x0 + (bar_px - text_w) // 2
        text_y = bar_y0 - TEXT_GAP - text_h

        draw = _IDraw.Draw(img)

        draw.rectangle([bar_x0 - 1, bar_y0 - 1, bar_x1 + 1, bar_y1 + 1],
                       fill=(0, 0, 0))
        draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(255, 255, 255))

        draw.text((text_x + 1, text_y + 1), bar_label, font=font,
                  fill=(0, 0, 0))
        draw.text((text_x, text_y), bar_label, font=font,
                  fill=(255, 255, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")
