"""
3D EM Augmentation Pipeline for OrganelleNet.

Provides standard augmentations for electron microscopy segmentation:
- Random 3D flips (all axes)
- Random 90° rotations (XY plane)
- Intensity scaling (brightness/contrast)
- Additive Gaussian noise
- Gamma correction
- Elastic deformation (coarse-grid displacement)

Plus a novel augmentation:
- Resolution augmentation: random downsample-then-upsample to simulate
  coarser acquisition, forcing scale-invariant feature learning.

All augmentations operate on numpy arrays:
  em_volume:    float32, shape [Z, Y, X], range [0, 1]
  label_volume: int64,   shape [Z, Y, X]
"""

import numpy as np
from scipy.ndimage import map_coordinates, zoom, gaussian_filter


def apply_augmentations(em_volume, label_volume, aug_config):
    """
    Apply the full augmentation pipeline to a 3D EM + label pair.

    Parameters
    ----------
    em_volume : np.ndarray
        Float32 array of shape [Z, Y, X], values in [0, 1].
    label_volume : np.ndarray
        Int64 array of shape [Z, Y, X], semantic class labels.
    aug_config : AugmentationConfig
        Configuration object controlling which augmentations are active.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Augmented (em_volume, label_volume).
    """
    if not aug_config.enabled:
        return em_volume, label_volume

    # ── 1. Geometric augmentations (applied to both EM and labels) ──────

    # Random 3D flips — independently along each axis
    if aug_config.flip:
        for axis in range(3):
            if np.random.random() < 0.5:
                em_volume = np.flip(em_volume, axis=axis).copy()
                label_volume = np.flip(label_volume, axis=axis).copy()

    # Random 90° rotation in the XY plane
    if aug_config.rotate90:
        k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
        if k > 0:
            em_volume = np.rot90(em_volume, k=k, axes=(1, 2)).copy()
            label_volume = np.rot90(label_volume, k=k, axes=(1, 2)).copy()

    # Elastic deformation
    if aug_config.elastic_enabled and np.random.random() < aug_config.elastic_prob:
        em_volume, label_volume = _elastic_deform_3d(
            em_volume, label_volume,
            alpha=aug_config.elastic_alpha,
            sigma=aug_config.elastic_sigma,
        )

    # ── 2. Resolution augmentation (EM only — novel contribution) ───────

    if (aug_config.resolution_aug_enabled
            and np.random.random() < aug_config.resolution_aug_prob):
        em_volume = _resolution_augment(
            em_volume,
            scale_range=aug_config.resolution_scale_range,
        )

    # ── 3. Intensity augmentations (EM only) ────────────────────────────

    # Brightness / contrast jitter
    scale = np.random.uniform(*aug_config.intensity_scale_range)
    shift = np.random.uniform(*aug_config.intensity_shift_range)
    em_volume = em_volume * scale + shift

    # Gamma correction
    gamma = np.random.uniform(*aug_config.gamma_range)
    em_volume = np.clip(em_volume, 1e-8, 1.0)  # avoid log(0)
    em_volume = np.power(em_volume, gamma)

    # Additive Gaussian noise
    if aug_config.gaussian_noise_std > 0:
        noise = np.random.normal(
            0, aug_config.gaussian_noise_std, em_volume.shape
        ).astype(np.float32)
        em_volume = em_volume + noise

    # Final clamp to valid range
    em_volume = np.clip(em_volume, 0.0, 1.0).astype(np.float32)

    return em_volume, label_volume


# ─── Internal helpers ───────────────────────────────────────────────────────


def _elastic_deform_3d(em_volume, label_volume, alpha=80.0, sigma=10.0):
    """
    Apply smooth elastic deformation to a 3D volume pair.

    Uses a coarse random displacement grid (4×4×4) that is upsampled and
    smoothed, then applied via scipy map_coordinates.

    Parameters
    ----------
    em_volume : np.ndarray, float32, [Z, Y, X]
    label_volume : np.ndarray, int64, [Z, Y, X]
    alpha : float
        Displacement magnitude scaling factor.
    sigma : float
        Gaussian smoothing sigma for the displacement field.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
    """
    shape = em_volume.shape
    coarse = (4, 4, 4)
    zoom_factors = [s / c for s, c in zip(shape, coarse)]

    # Random displacement on a coarse grid, upsampled smoothly
    dz = zoom(np.random.randn(*coarse).astype(np.float32), zoom_factors, order=3) * alpha
    dy = zoom(np.random.randn(*coarse).astype(np.float32), zoom_factors, order=3) * alpha
    dx = zoom(np.random.randn(*coarse).astype(np.float32), zoom_factors, order=3) * alpha

    # Smooth the displacement field
    dz = gaussian_filter(dz, sigma)
    dy = gaussian_filter(dy, sigma)
    dx = gaussian_filter(dx, sigma)

    # Build displaced coordinate grids
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    coords = [
        np.clip(z + dz, 0, shape[0] - 1),
        np.clip(y + dy, 0, shape[1] - 1),
        np.clip(x + dx, 0, shape[2] - 1),
    ]

    # Interpolate: trilinear for EM, nearest-neighbor for labels
    em_out = map_coordinates(em_volume, coords, order=1, mode="reflect").astype(np.float32)
    lbl_out = map_coordinates(
        label_volume.astype(np.float64), coords, order=0, mode="reflect"
    ).astype(np.int64)

    return em_out, lbl_out


def _resolution_augment(em_volume, scale_range=(0.25, 1.0)):
    """
    Simulate coarser acquisition by downsampling then upsampling.

    This forces the model to learn scale-invariant features, bridging the
    performance gap between fine (2–4 nm) and coarse (16–64 nm) resolution
    bands in the CellMap dataset.

    Labels are NOT modified — the model must still predict at full resolution
    from degraded input, learning to "see through" the blur.

    Parameters
    ----------
    em_volume : np.ndarray, float32, [Z, Y, X]
    scale_range : tuple of (float, float)
        (min_scale, max_scale). scale < 1 means coarser resolution.
        E.g. 0.25 = 4× coarser, 0.5 = 2× coarser.

    Returns
    -------
    np.ndarray, float32
    """
    original_shape = em_volume.shape
    scale = np.random.uniform(*scale_range)

    # Skip if scale is very close to 1 (negligible effect)
    if scale >= 0.95:
        return em_volume

    # Downsample with trilinear interpolation
    downsampled = zoom(em_volume, scale, order=1)

    # Upsample back to original resolution
    upsample_factors = [o / d for o, d in zip(original_shape, downsampled.shape)]
    upsampled = zoom(downsampled, upsample_factors, order=1)

    # zoom() can produce shapes off by 1; ensure exact match
    if upsampled.shape != original_shape:
        result = np.zeros(original_shape, dtype=np.float32)
        slices = tuple(
            slice(0, min(u, o))
            for u, o in zip(upsampled.shape, original_shape)
        )
        result[slices] = upsampled[slices]
        return result

    return upsampled.astype(np.float32)
