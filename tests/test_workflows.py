import numpy as np
import SimpleITK as sitk

from voidspace import VoidspaceParameters, analyze_maps, compare, intersect_masks, run_case
import pytest


@pytest.mark.parametrize("size, origin", [(3, (1, 1, 1)), (5, (1, 0, 0))])
def test_run_case_places_analysis_mask_on_segmentation_grid(tmp_path, size, origin):
    segmentation = np.zeros((5, 5, 5), dtype=np.uint8)
    _write_mask(tmp_path / "seg.nii.gz", segmentation)
    mask_image = sitk.GetImageFromArray(np.ones((size,) * 3, dtype=np.uint8))
    mask_image.SetOrigin(origin)
    sitk.WriteImage(mask_image, str(tmp_path / "domain.nii.gz"))
    result = run_case(segmentation_path=tmp_path / "seg.nii.gz",
                      mask_path=tmp_path / "domain.nii.gz", output_dir=tmp_path / "out",
                      parameters=VoidspaceParameters(closing_radius_mm=0, boundary_erosion_radius_mm=0,
                          min_large_void_volume_mm3=0, bone_speckle_min_voxels=1, void_speckle_min_voxels=1))
    assert result.metrics.total_volume_mm3 == (27 if size == 3 else 100)
    output = sitk.GetArrayFromImage(sitk.ReadImage(str(result.all_mask_path)))
    assert not output[:, :, 0].any()


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


@pytest.mark.parametrize("shift", [0.0, 0.0607])
@pytest.mark.parametrize("source_spacing, common_spacing", [
    ((0.0607,) * 3, (float(np.float32(0.0607)),) * 3),
    ((0.06069900095462799, 0.06069900095462799, 0.06069599837064743),
     (0.06069965288043022, 0.06069965288043022, 0.06069643050432205)),
])
def test_intersect_masks_accepts_header_rounding_and_preserves_physical_placement(
    tmp_path, shift, source_spacing, common_spacing,
):
    full = sitk.GetImageFromArray(np.ones((5, 5, 5), dtype=np.uint8))
    full.SetSpacing(source_spacing)
    sitk.WriteImage(full, str(tmp_path / "full.mha"))
    common = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8))
    common.SetSpacing(common_spacing)
    common.SetOrigin((shift,) * 3)
    sitk.WriteImage(common, str(tmp_path / "common.nii.gz"))

    output = intersect_masks(
        [tmp_path / "full.mha", tmp_path / "common.nii.gz"],
        output_path=tmp_path / "intersection.nii.gz",
    )
    image = sitk.ReadImage(str(output))
    expected = np.zeros((5, 5, 5), dtype=bool)
    start = 0 if shift == 0 else 1
    expected[start:start + 3, start:start + 3, start:start + 3] = True
    assert np.array_equal(sitk.GetArrayFromImage(image).astype(bool), expected)
    assert np.allclose(image.GetSpacing(), source_spacing, rtol=0, atol=1e-8)


def test_intersect_masks_reports_genuinely_different_resolution(tmp_path):
    for name, spacing in [("full", 0.0607), ("common", 0.082)]:
        image = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8))
        image.SetSpacing((spacing,) * 3)
        sitk.WriteImage(image, str(tmp_path / f"{name}.mha"))
    with pytest.raises(ValueError, match="spacing.*full.*common"):
        intersect_masks(
            [tmp_path / "full.mha", tmp_path / "common.mha"],
            output_path=tmp_path / "intersection.nii.gz",
        )


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


def test_analyze_maps_honors_direction_with_matching_sizes_and_origins(tmp_path):
    full = np.ones((5, 5, 5), dtype=bool)
    _write_mask(tmp_path / "all.nii.gz", full)
    _write_mask(tmp_path / "large.nii.gz", full)
    mask = sitk.GetImageFromArray(full.astype(np.uint8))
    mask.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))
    sitk.WriteImage(mask, str(tmp_path / "domain.nii.gz"))
    result = analyze_maps(large_void_path=tmp_path / "large.nii.gz", all_void_path=tmp_path / "all.nii.gz",
                          mask_path=tmp_path / "domain.nii.gz", output_dir=tmp_path / "out")
    assert result.metrics.total_volume_mm3 == 25


def test_compare_places_cropped_maps_in_same_physical_space(tmp_path):
    baseline = np.zeros((5, 5, 5), dtype=bool)
    baseline[2, 2, 2] = True
    _write_mask(tmp_path / "baseline.nii.gz", baseline)
    cropped = sitk.GetImageFromArray(np.ones((1, 1, 1), dtype=np.uint8))
    cropped.SetOrigin((2, 2, 2))
    sitk.WriteImage(cropped, str(tmp_path / "followup.nii.gz"))
    result = compare(baseline_void_path=tmp_path / "baseline.nii.gz",
                     followup_void_path=tmp_path / "followup.nii.gz", output_dir=tmp_path / "out")
    assert result.metrics.stable_volume_mm3 == 1
    assert result.metrics.expanded_volume_mm3 == result.metrics.contracted_volume_mm3 == 0
