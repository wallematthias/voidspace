from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from voidspace.change import classify_voidspace_change, measure_voidspace_change
from voidspace.io import AimReference, read_mask, write_mask_like, write_metrics_csv
from voidspace.metrics import measure_voidspace
from voidspace.models import VoidspaceCompareResult, VoidspaceParameters, VoidspaceRunResult
from voidspace.segment import segment_voidspace


def _metric_row(metrics, **context):
    return {
        **context,
        "VS.TV": metrics.vstv_percent,
        "VS.V": metrics.volume_mm3,
        "Tt.V": metrics.total_volume_mm3,
        "VS.N": metrics.component_count,
        "VS.Ar": metrics.projected_area_mm2,
    }


def _mask_extension(reference) -> str:
    return ".AIM" if isinstance(reference, AimReference) else ".nii.gz"


def _reference_image(reference) -> sitk.Image:
    return reference.image if isinstance(reference, AimReference) else reference


def _array_on_reference_grid(array, source_reference, target_reference) -> np.ndarray:
    source_image = _reference_image(source_reference)
    target_image = _reference_image(target_reference)
    if source_image.GetSize() == target_image.GetSize() and all(
        np.allclose(getattr(source_image, getter)(), getattr(target_image, getter)(), rtol=0, atol=1e-6)
        for getter in ("GetOrigin", "GetSpacing", "GetDirection")
    ):
        return np.asarray(array, dtype=bool)
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))
    image.CopyInformation(source_image)
    resampled = sitk.Resample(
        image,
        target_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return sitk.GetArrayFromImage(resampled).astype(bool)


def intersect_masks(
    mask_paths: list[Path | str] | tuple[Path | str, ...],
    *,
    output_path: Path | str,
    force: bool = False,
) -> Path:
    """Write the voxelwise intersection of masks that share one image space."""
    paths = [Path(path) for path in mask_paths]
    if len(paths) < 2:
        raise ValueError("at least two masks are required")
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")

    intersection, spacing, reference = read_mask(paths[0])
    for path in paths[1:]:
        mask, mask_spacing, mask_reference = read_mask(path)
        if mask_spacing != spacing:
            raise ValueError("all masks must have the same spacing")
        mask = _array_on_reference_grid(mask, mask_reference, reference)
        if mask.shape != intersection.shape:
            raise ValueError("all masks must overlap the reference image space")
        intersection = intersection & mask

    write_mask_like(intersection, reference, output_path)
    return output_path


def run_case(
    *,
    segmentation_path: Path | str,
    output_dir: Path | str,
    mask_path: Path | str | None = None,
    parameters: VoidspaceParameters | None = None,
    force: bool = False,
) -> VoidspaceRunResult:
    """Run cross-sectional voidspace segmentation and measurement for one scan."""
    output_dir = Path(output_dir)
    segmentation, spacing, reference = read_mask(segmentation_path)
    mask_extension = _mask_extension(reference)
    large_path = output_dir / f"voidspace_large_mask{mask_extension}"
    all_path = output_dir / f"voidspace_all_mask{mask_extension}"
    measurements_path = output_dir / "voidspace_measurements.csv"

    existing_outputs = [path for path in (large_path, all_path, measurements_path) if path.exists()]
    if existing_outputs and not force:
        raise FileExistsError(f"output already exists: {existing_outputs[0]}")

    if mask_path is not None:
        domain_mask, _mask_spacing, mask_reference = read_mask(mask_path)
        domain_mask = _array_on_reference_grid(domain_mask, mask_reference, reference)
    else:
        domain_mask = None

    params = parameters or VoidspaceParameters.xtremectii_defaults()
    masks = segment_voidspace(segmentation, spacing, mask=domain_mask, parameters=params)
    write_mask_like(masks.large_void, reference, large_path)
    write_mask_like(masks.all_void, reference, all_path)

    total_mask = domain_mask if domain_mask is not None else (masks.filled_bone | masks.all_void)
    metrics = measure_voidspace(masks.large_void, total_mask, spacing, connectivity=params.connectivity)
    write_metrics_csv(measurements_path, [_metric_row(metrics)])

    return VoidspaceRunResult(large_path, all_path, measurements_path, metrics)


def analyze_maps(
    *,
    large_void_path: Path | str,
    all_void_path: Path | str,
    mask_path: Path | str,
    output_dir: Path | str,
    force: bool = False,
    connectivity: int = 3,
) -> VoidspaceRunResult:
    """Measure existing voidspace maps within an analysis mask without recomputing maps."""
    output_dir = Path(output_dir)
    large_void, spacing, reference = read_mask(large_void_path)
    mask_extension = _mask_extension(reference)
    large_path = output_dir / f"voidspace_large_mask{mask_extension}"
    all_path = output_dir / f"voidspace_all_mask{mask_extension}"
    measurements_path = output_dir / "voidspace_measurements.csv"

    existing_outputs = [path for path in (large_path, all_path, measurements_path) if path.exists()]
    if existing_outputs and not force:
        raise FileExistsError(f"output already exists: {existing_outputs[0]}")

    all_void, _all_spacing, all_reference = read_mask(all_void_path)
    all_void = _array_on_reference_grid(all_void, all_reference, reference)
    domain_mask, _mask_spacing, mask_reference = read_mask(mask_path)
    domain_mask = _array_on_reference_grid(domain_mask, mask_reference, reference)

    masked_large_void = large_void & domain_mask
    masked_all_void = all_void & domain_mask
    write_mask_like(masked_large_void, reference, large_path)
    write_mask_like(masked_all_void, reference, all_path)

    metrics = measure_voidspace(masked_large_void, domain_mask, spacing, connectivity=connectivity)
    write_metrics_csv(measurements_path, [_metric_row(metrics)])

    return VoidspaceRunResult(large_path, all_path, measurements_path, metrics)


def compare(
    *,
    baseline_void_path: Path | str,
    followup_void_path: Path | str,
    output_dir: Path | str,
    mask_path: Path | str | None = None,
    force: bool = False,
) -> VoidspaceCompareResult:
    """Compare already aligned baseline and follow-up voidspace masks."""
    output_dir = Path(output_dir)
    baseline, spacing, reference = read_mask(baseline_void_path)
    mask_extension = _mask_extension(reference)
    stable_path = output_dir / f"voidspace_quiescent_mask{mask_extension}"
    expanded_path = output_dir / f"voidspace_expanded_mask{mask_extension}"
    contracted_path = output_dir / f"voidspace_contracted_mask{mask_extension}"
    measurements_path = output_dir / "voidspace_change_measurements.csv"
    existing_outputs = [
        path for path in (stable_path, expanded_path, contracted_path, measurements_path) if path.exists()
    ]
    if existing_outputs and not force:
        raise FileExistsError(f"output already exists: {existing_outputs[0]}")

    followup, _followup_spacing, followup_reference = read_mask(followup_void_path)
    followup = _array_on_reference_grid(followup, followup_reference, reference)

    domain_mask = None
    if mask_path is not None:
        domain_mask, _mask_spacing, mask_reference = read_mask(mask_path)
        domain_mask = _array_on_reference_grid(domain_mask, mask_reference, reference)

    change = classify_voidspace_change(baseline, followup, mask=domain_mask)
    write_mask_like(change.stable, reference, stable_path)
    write_mask_like(change.expanded, reference, expanded_path)
    write_mask_like(change.contracted, reference, contracted_path)
    metrics = measure_voidspace_change(change, spacing)
    write_metrics_csv(
        measurements_path,
        [
            {
                "stable.VS.V": metrics.stable_volume_mm3,
                "quiescent.VS.V": metrics.stable_volume_mm3,
                "expanded.VS.V": metrics.expanded_volume_mm3,
                "contracted.VS.V": metrics.contracted_volume_mm3,
                "net.VS.V": metrics.net_change_volume_mm3,
            }
        ],
    )
    return VoidspaceCompareResult(
        stable_path,
        expanded_path,
        contracted_path,
        measurements_path,
        metrics,
    )


run_voidspace_case = run_case
run_voidspace_change_case = compare
