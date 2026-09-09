import numpy as np
import SimpleITK as sitk

from voidspace import VoidspaceParameters, analyze_maps, compare, intersect_masks, run_case


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


def test_analyze_maps_reuses_existing_void_masks_and_measures_inside_mask(tmp_path):
    all_void = np.ones((3, 3, 3), dtype=bool)
    large_void = np.zeros_like(all_void)
    large_void[0, 0, 0] = True
    large_void[2, 2, 2] = True
    mask = np.zeros_like(all_void)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True
    _write_mask(tmp_path / "all.nii.gz", all_void)
    _write_mask(tmp_path / "large.nii.gz", large_void)
    _write_mask(tmp_path / "mask.nii.gz", mask)

    result = analyze_maps(
        large_void_path=tmp_path / "large.nii.gz",
        all_void_path=tmp_path / "all.nii.gz",
        mask_path=tmp_path / "mask.nii.gz",
        output_dir=tmp_path / "masked",
    )

    assert result.large_mask_path.is_file()
    assert result.all_mask_path.is_file()
    assert result.measurements_path.is_file()
    assert result.metrics.volume_mm3 == 1.0
    assert result.metrics.total_volume_mm3 == 2.0


def test_intersect_masks_writes_boolean_intersection(tmp_path):
    full = np.ones((3, 3, 3), dtype=bool)
    full[2, :, :] = False
    common = np.zeros_like(full)
    common[:, 1:, :] = True
    _write_mask(tmp_path / "full.nii.gz", full)
    _write_mask(tmp_path / "common.nii.gz", common)

    output_path = intersect_masks(
        [tmp_path / "full.nii.gz", tmp_path / "common.nii.gz"],
        output_path=tmp_path / "analysis_mask.nii.gz",
    )

    image = sitk.ReadImage(str(output_path))
    result = sitk.GetArrayFromImage(image).astype(bool)
    assert result.sum() == 12
    assert np.array_equal(result, full & common)


def test_intersect_masks_resamples_cropped_mask_to_reference_grid(tmp_path):
    full = np.ones((5, 5, 5), dtype=bool)
    common = np.ones((3, 3, 3), dtype=bool)
    _write_mask(tmp_path / "full.nii.gz", full)
    common_image = sitk.GetImageFromArray(common.astype("uint8"))
    common_image.SetSpacing((1.0, 1.0, 1.0))
    common_image.SetOrigin((1.0, 1.0, 1.0))
    sitk.WriteImage(common_image, str(tmp_path / "common.nii.gz"))

    output_path = intersect_masks(
        [tmp_path / "full.nii.gz", tmp_path / "common.nii.gz"],
        output_path=tmp_path / "analysis_mask.nii.gz",
    )

    image = sitk.ReadImage(str(output_path))
    result = sitk.GetArrayFromImage(image).astype(bool)
    expected = np.zeros_like(full)
    expected[1:4, 1:4, 1:4] = True
    assert image.GetSize() == (5, 5, 5)
    assert np.array_equal(result, expected)
