from __future__ import annotations

from pathlib import Path

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
        domain_mask, mask_spacing, _mask_reference = read_mask(mask_path)
        if mask_spacing != spacing or domain_mask.shape != segmentation.shape:
            raise ValueError("mask must be in the same space as segmentation")
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
    stable_path = output_dir / f"voidspace_stable_mask{mask_extension}"
    expanded_path = output_dir / f"voidspace_expanded_mask{mask_extension}"
    contracted_path = output_dir / f"voidspace_contracted_mask{mask_extension}"
    measurements_path = output_dir / "voidspace_change_measurements.csv"
    existing_outputs = [
        path for path in (stable_path, expanded_path, contracted_path, measurements_path) if path.exists()
    ]
    if existing_outputs and not force:
        raise FileExistsError(f"output already exists: {existing_outputs[0]}")

    followup, followup_spacing, _followup_reference = read_mask(followup_void_path)
    if followup_spacing != spacing or followup.shape != baseline.shape:
        raise ValueError("baseline and followup void masks must already be aligned")

    domain_mask = None
    if mask_path is not None:
        domain_mask, mask_spacing, _mask_reference = read_mask(mask_path)
        if mask_spacing != spacing or domain_mask.shape != baseline.shape:
            raise ValueError("mask must be aligned with the void masks")

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
