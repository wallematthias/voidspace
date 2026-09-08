from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from voidspace.models import VoidspaceMetrics
from voidspace.morphology import connectivity_structure, voxel_volume_mm3


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"all masks must have the same shape, got {sorted(shapes)}")


def measure_voidspace(
    void_mask: np.ndarray,
    total_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    mask: np.ndarray | None = None,
    connectivity: int = 3,
) -> VoidspaceMetrics:
    void_mask = np.asarray(void_mask, dtype=bool)
    total_mask = np.asarray(total_mask, dtype=bool)
    if void_mask.ndim != 3 or total_mask.ndim != 3:
        raise ValueError("void_mask and total_mask must be 3D arrays")
    _validate_same_shape(void_mask, total_mask)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        _validate_same_shape(void_mask, mask)
        total = total_mask & mask
    else:
        total = total_mask

    void = void_mask & total
    voxel_volume = voxel_volume_mm3(spacing_mm)
    void_volume = float(void.sum() * voxel_volume)
    total_volume = float(total.sum() * voxel_volume)
    _labels, component_count = ndi.label(void, structure=connectivity_structure(connectivity))
    projected_voxels = int(np.any(void, axis=0).sum())
    projected_area = float(projected_voxels * float(spacing_mm[1]) * float(spacing_mm[2]))
    vstv = 0.0 if total_volume == 0 else float(100.0 * void_volume / total_volume)

    return VoidspaceMetrics(
        volume_mm3=void_volume,
        total_volume_mm3=total_volume,
        vstv_percent=vstv,
        component_count=int(component_count),
        projected_area_mm2=projected_area,
    )
