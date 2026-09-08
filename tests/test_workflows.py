import numpy as np
import SimpleITK as sitk

from voidspace import VoidspaceParameters, compare, run_case


def _write_mask(path, array):
    image = sitk.GetImageFromArray(array.astype("uint8"))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def test_run_case_returns_metrics_and_output_paths(tmp_path):
    segmentation = np.ones((9, 9, 9), dtype=bool)
    segmentation[4, 4, 4] = False
    _write_mask(tmp_path / "seg.nii.gz", segmentation)

    result = run_case(
        segmentation_path=tmp_path / "seg.nii.gz",
        output_dir=tmp_path / "out",
        parameters=VoidspaceParameters(
            closing_radius_mm=0.0,
            boundary_erosion_radius_mm=0.0,
            min_large_void_volume_mm3=0.0,
            bone_speckle_min_voxels=1,
            void_speckle_min_voxels=1,
        ),
    )

    assert result.large_mask_path.is_file()
    assert result.all_mask_path.is_file()
    assert result.measurements_path.is_file()
    assert result.metrics.component_count == 1


def test_compare_returns_change_metrics_and_output_paths(tmp_path):
    baseline = np.zeros((2, 2, 2), dtype=bool)
    followup = np.zeros_like(baseline)
    baseline[0, 0, 0] = True
    followup[1, 1, 1] = True
    _write_mask(tmp_path / "baseline.nii.gz", baseline)
    _write_mask(tmp_path / "followup.nii.gz", followup)

    result = compare(
        baseline_void_path=tmp_path / "baseline.nii.gz",
        followup_void_path=tmp_path / "followup.nii.gz",
        output_dir=tmp_path / "out",
    )

    assert result.expanded_mask_path.is_file()
    assert result.contracted_mask_path.is_file()
    assert result.measurements_path.is_file()
    assert result.metrics.expanded_volume_mm3 == 1.0
    assert result.metrics.contracted_volume_mm3 == 1.0
