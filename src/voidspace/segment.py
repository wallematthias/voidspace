from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from voidspace.models import VoidspaceMasks, VoidspaceParameters
from voidspace.morphology import (
    ellipsoid_footprint,
    min_voxels_for_volume,
    remove_border_connected_components,
    remove_small_components,
)


def _validate_3d(name: str, array: np.ndarray) -> np.ndarray:
    mask = np.asarray(array, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")
    return mask


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"all masks must have the same shape, got {sorted(shapes)}")


def segment_voidspace(
    segmentation: np.ndarray,
    periosteal_mask: np.ndarray | None,
    spacing_mm: tuple[float, float, float],
    parameters: VoidspaceParameters | None = None,
) -> VoidspaceMasks:
    params = parameters or VoidspaceParameters.xtremectii_defaults()
    bone = _validate_3d("segmentation", segmentation)
    if periosteal_mask is None:
        domain = np.ones_like(bone, dtype=bool)
        domain_source = "segmentation_border_background"
    else:
        domain = _validate_3d("periosteal_mask", periosteal_mask)
        _validate_same_shape(bone, domain)
        domain_source = "periosteal_mask"

    bone_in_domain = remove_small_components(
        bone & domain,
        params.bone_speckle_min_voxels,
        connectivity=params.connectivity,
    )
    closing_fp = ellipsoid_footprint(params.closing_radius_mm, spacing_mm)
    filled_bone = ndi.binary_closing(bone_in_domain, structure=closing_fp) & domain
    if periosteal_mask is None:
        interior_background = remove_border_connected_components(
            ~filled_bone,
            connectivity=params.connectivity,
        )
        domain = filled_bone | interior_background
        filled_bone = filled_bone & domain
    candidate_void = domain & ~filled_bone

    if params.boundary_erosion_radius_mm > 0:
        erosion_fp = ellipsoid_footprint(params.boundary_erosion_radius_mm, spacing_mm)
        candidate_void = ndi.binary_erosion(candidate_void, structure=erosion_fp, border_value=0)
    candidate_void = candidate_void & domain

    all_void = remove_small_components(
        candidate_void,
        params.void_speckle_min_voxels,
        connectivity=params.connectivity,
    )
    large_min_voxels = min_voxels_for_volume(params.min_large_void_volume_mm3, spacing_mm)
    large_void = remove_small_components(all_void, large_min_voxels, connectivity=params.connectivity)

    return VoidspaceMasks(
        all_void=all_void,
        large_void=large_void,
        filled_bone=filled_bone,
        metadata={
            "spacing_mm": tuple(float(v) for v in spacing_mm),
            "closing_radius_mm": params.closing_radius_mm,
            "boundary_erosion_radius_mm": params.boundary_erosion_radius_mm,
            "min_large_void_volume_mm3": params.min_large_void_volume_mm3,
            "structuring_element": "spacing_aware_ellipsoid",
            "domain_source": domain_source,
        },
    )
