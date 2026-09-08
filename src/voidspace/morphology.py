from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def _spacing_array(spacing_mm: tuple[float, float, float]) -> np.ndarray:
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,):
        raise ValueError("spacing_mm must contain exactly three values")
    if np.any(spacing <= 0):
        raise ValueError("spacing_mm values must be > 0")
    return spacing


def voxel_volume_mm3(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(_spacing_array(spacing_mm)))


def min_voxels_for_volume(volume_mm3: float, spacing_mm: tuple[float, float, float]) -> int:
    if volume_mm3 < 0:
        raise ValueError("volume_mm3 must be >= 0")
    return int(math.ceil(float(volume_mm3) / voxel_volume_mm3(spacing_mm)))


def ellipsoid_footprint(radius_mm: float, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    if radius_mm < 0:
        raise ValueError("radius_mm must be >= 0")
    spacing = _spacing_array(spacing_mm)
    radii_voxels = np.ceil(radius_mm / spacing).astype(int)
    if radius_mm == 0:
        return np.ones(tuple(2 * radii_voxels + 1), dtype=bool)

    grids = np.ogrid[
        -radii_voxels[0] : radii_voxels[0] + 1,
        -radii_voxels[1] : radii_voxels[1] + 1,
        -radii_voxels[2] : radii_voxels[2] + 1,
    ]
    distance2 = np.zeros(tuple(2 * radii_voxels + 1), dtype=float)
    for axis_grid, axis_spacing in zip(grids, spacing, strict=True):
        distance2 = distance2 + ((axis_grid * axis_spacing) / radius_mm) ** 2
    return distance2 <= 1.0


def connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2, or 3")
    return ndi.generate_binary_structure(rank=3, connectivity=connectivity)


def remove_small_components(mask: np.ndarray, min_voxels: int, connectivity: int = 3) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D array")
    if min_voxels <= 1:
        return mask.copy()

    labels, count = ndi.label(mask, structure=connectivity_structure(connectivity))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_voxels)
    keep[0] = False
    return keep[labels]


def remove_border_connected_components(mask: np.ndarray, connectivity: int = 3) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D array")

    labels, count = ndi.label(mask, structure=connectivity_structure(connectivity))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    border_labels = set(np.unique(labels[0, :, :]))
    border_labels.update(np.unique(labels[-1, :, :]))
    border_labels.update(np.unique(labels[:, 0, :]))
    border_labels.update(np.unique(labels[:, -1, :]))
    border_labels.update(np.unique(labels[:, :, 0]))
    border_labels.update(np.unique(labels[:, :, -1]))
    border_labels.discard(0)
    remove = np.zeros(count + 1, dtype=bool)
    if border_labels:
        remove[list(border_labels)] = True
    return mask & ~remove[labels]
