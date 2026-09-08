from __future__ import annotations

import numpy as np

from voidspace.models import VoidspaceChangeMasks, VoidspaceChangeMetrics
from voidspace.morphology import voxel_volume_mm3


def classify_voidspace_change(
    baseline_void: np.ndarray,
    followup_void: np.ndarray,
    analysis_mask: np.ndarray | None = None,
) -> VoidspaceChangeMasks:
    baseline = np.asarray(baseline_void, dtype=bool)
    followup = np.asarray(followup_void, dtype=bool)
    if baseline.shape != followup.shape:
        raise ValueError("baseline_void and followup_void must have the same shape")
    if analysis_mask is not None:
        mask = np.asarray(analysis_mask, dtype=bool)
        if mask.shape != baseline.shape:
            raise ValueError("analysis_mask must match void mask shape")
        baseline = baseline & mask
        followup = followup & mask

    return VoidspaceChangeMasks(
        stable=baseline & followup,
        expanded=followup & ~baseline,
        contracted=baseline & ~followup,
    )


def measure_voidspace_change(
    change: VoidspaceChangeMasks,
    spacing_mm: tuple[float, float, float],
) -> VoidspaceChangeMetrics:
    voxel_volume = voxel_volume_mm3(spacing_mm)
    stable = float(np.asarray(change.stable, dtype=bool).sum() * voxel_volume)
    expanded = float(np.asarray(change.expanded, dtype=bool).sum() * voxel_volume)
    contracted = float(np.asarray(change.contracted, dtype=bool).sum() * voxel_volume)
    return VoidspaceChangeMetrics(
        stable_volume_mm3=stable,
        expanded_volume_mm3=expanded,
        contracted_volume_mm3=contracted,
        net_change_volume_mm3=expanded - contracted,
    )

