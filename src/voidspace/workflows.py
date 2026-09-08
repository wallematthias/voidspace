from __future__ import annotations

from pathlib import Path

import numpy as np

from voidspace import (
    VoidspaceParameters,
    classify_voidspace_change,
    measure_voidspace,
    measure_voidspace_change,
    segment_voidspace,
)
from voidspace.io import read_mask, write_mask_like, write_metrics_csv


def _metric_row(metrics, **context):
    return {
        **context,
        "VS.TV": metrics.vstv_percent,
        "VS.V": metrics.volume_mm3,
        "Tt.V": metrics.total_volume_mm3,
        "VS.N": metrics.component_count,
        "VS.Ar": metrics.projected_area_mm2,
        "analysis_masked": metrics.metadata.get("analysis_masked", False),
    }


def _parameters_from_kwargs(**kwargs) -> VoidspaceParameters:
    defaults = VoidspaceParameters.xtremectii_defaults()
    values = {
        "closing_radius_mm": kwargs.get("closing_radius_mm", defaults.closing_radius_mm),
        "boundary_erosion_radius_mm": kwargs.get(
            "boundary_erosion_radius_mm", defaults.boundary_erosion_radius_mm
        ),
        "min_large_void_volume_mm3": kwargs.get(
            "min_large_void_volume_mm3", defaults.min_large_void_volume_mm3
        ),
        "bone_speckle_min_voxels": kwargs.get(
            "bone_speckle_min_voxels", defaults.bone_speckle_min_voxels
        ),
        "void_speckle_min_voxels": kwargs.get(
            "void_speckle_min_voxels", defaults.void_speckle_min_voxels
        ),
        "connectivity": kwargs.get("connectivity", defaults.connectivity),
    }
    return VoidspaceParameters(**values)


def run_voidspace_case(
    *,
    segmentation_path: Path | str,
    output_dir: Path | str,
    analysis_mask_path: Path | str | None = None,
    periosteal_mask_path: Path | str | None = None,
    subject_id: str = "",
    session_id: str = "",
    site: str = "",
    space: str = "native",
    reference_session_id: str = "",
    parameters: VoidspaceParameters | None = None,
    force: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    large_path = output_dir / "voidspace_large_mask.nii.gz"
    all_path = output_dir / "voidspace_all_mask.nii.gz"
    measurements_path = output_dir / "voidspace_measurements.csv"
    masked_measurements_path = output_dir / "voidspace_analysis_masked_measurements.csv"

    outputs = {
        "large_mask": large_path,
        "all_mask": all_path,
        "measurements": measurements_path,
    }
    if analysis_mask_path is not None:
        outputs["analysis_masked_measurements"] = masked_measurements_path
    if measurements_path.exists() and large_path.exists() and not force:
        return outputs

    segmentation, spacing, reference = read_mask(segmentation_path)
    if periosteal_mask_path is not None:
        periosteal_mask, peri_spacing, _peri_reference = read_mask(periosteal_mask_path)
        if peri_spacing != spacing or periosteal_mask.shape != segmentation.shape:
            raise ValueError("periosteal mask must be in the same space as segmentation")
    else:
        periosteal_mask = np.ones_like(segmentation, dtype=bool)

    params = parameters or VoidspaceParameters.xtremectii_defaults()
    masks = segment_voidspace(segmentation, periosteal_mask, spacing, params)
    write_mask_like(masks.large_void, reference, large_path)
    write_mask_like(masks.all_void, reference, all_path)

    context = {
        "subject_id": subject_id,
        "session_id": session_id,
        "site": site,
        "space": space,
        "reference_session_id": reference_session_id,
    }
    metrics = measure_voidspace(masks.large_void, periosteal_mask, spacing, connectivity=params.connectivity)
    write_metrics_csv(measurements_path, [_metric_row(metrics, **context)])

    if analysis_mask_path is not None:
        analysis_mask, mask_spacing, _mask_reference = read_mask(analysis_mask_path)
        if mask_spacing != spacing or analysis_mask.shape != segmentation.shape:
            raise ValueError("analysis mask must be in the same space as segmentation")
        masked_metrics = measure_voidspace(
            masks.large_void,
            periosteal_mask,
            spacing,
            analysis_mask=analysis_mask,
            connectivity=params.connectivity,
        )
        write_metrics_csv(masked_measurements_path, [_metric_row(masked_metrics, **context)])

    return outputs


def run_voidspace_change_case(
    *,
    baseline_void_path: Path | str,
    followup_void_path: Path | str,
    output_dir: Path | str,
    analysis_mask_path: Path | str | None = None,
    subject_id: str = "",
    site: str = "",
    baseline_session_id: str = "",
    followup_session_id: str = "",
    force: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    stable_path = output_dir / "voidspace_stable_mask.nii.gz"
    expanded_path = output_dir / "voidspace_expanded_mask.nii.gz"
    contracted_path = output_dir / "voidspace_contracted_mask.nii.gz"
    measurements_path = output_dir / "voidspace_change_measurements.csv"
    outputs = {
        "stable_mask": stable_path,
        "expanded_mask": expanded_path,
        "contracted_mask": contracted_path,
        "change_measurements": measurements_path,
    }
    if measurements_path.exists() and expanded_path.exists() and not force:
        return outputs

    baseline, spacing, reference = read_mask(baseline_void_path)
    followup, followup_spacing, _followup_reference = read_mask(followup_void_path)
    if followup_spacing != spacing or followup.shape != baseline.shape:
        raise ValueError("baseline and followup void masks must already be aligned")

    analysis_mask = None
    if analysis_mask_path is not None:
        analysis_mask, mask_spacing, _mask_reference = read_mask(analysis_mask_path)
        if mask_spacing != spacing or analysis_mask.shape != baseline.shape:
            raise ValueError("analysis mask must be aligned with the void masks")

    change = classify_voidspace_change(baseline, followup, analysis_mask=analysis_mask)
    write_mask_like(change.stable, reference, stable_path)
    write_mask_like(change.expanded, reference, expanded_path)
    write_mask_like(change.contracted, reference, contracted_path)
    metrics = measure_voidspace_change(change, spacing)
    write_metrics_csv(
        measurements_path,
        [
            {
                "subject_id": subject_id,
                "site": site,
                "baseline_session_id": baseline_session_id,
                "followup_session_id": followup_session_id,
                "stable.VS.V": metrics.stable_volume_mm3,
                "expanded.VS.V": metrics.expanded_volume_mm3,
                "contracted.VS.V": metrics.contracted_volume_mm3,
                "net.VS.V": metrics.net_change_volume_mm3,
            }
        ],
    )
    return outputs

