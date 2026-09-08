from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np


def _validate_non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class VoidspaceParameters:
    closing_radius_mm: float = 0.738
    boundary_erosion_radius_mm: float = 0.366
    min_large_void_volume_mm3: float = 16.5
    bone_speckle_min_voxels: int = 6
    void_speckle_min_voxels: int = 6
    connectivity: int = 3

    def __post_init__(self) -> None:
        for name in (
            "closing_radius_mm",
            "boundary_erosion_radius_mm",
            "min_large_void_volume_mm3",
            "bone_speckle_min_voxels",
            "void_speckle_min_voxels",
        ):
            _validate_non_negative(name, getattr(self, name))
        if self.connectivity not in (1, 2, 3):
            raise ValueError("connectivity must be 1, 2, or 3")

    @classmethod
    def xtremectii_defaults(cls) -> "VoidspaceParameters":
        return cls()


@dataclass(frozen=True)
class VoidspaceMasks:
    all_void: np.ndarray
    large_void: np.ndarray
    filled_bone: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VoidspaceMetrics:
    volume_mm3: float
    total_volume_mm3: float
    vstv_percent: float
    component_count: int
    projected_area_mm2: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VoidspaceChangeMasks:
    stable: np.ndarray
    expanded: np.ndarray
    contracted: np.ndarray


@dataclass(frozen=True)
class VoidspaceChangeMetrics:
    stable_volume_mm3: float
    expanded_volume_mm3: float
    contracted_volume_mm3: float
    net_change_volume_mm3: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VoidspaceRunResult:
    large_mask_path: Path
    all_mask_path: Path
    measurements_path: Path
    metrics: VoidspaceMetrics


@dataclass(frozen=True)
class VoidspaceCompareResult:
    stable_mask_path: Path
    expanded_mask_path: Path
    contracted_mask_path: Path
    measurements_path: Path
    metrics: VoidspaceChangeMetrics
