import numpy as np

from voidspace import VoidspaceParameters, segment_voidspace


def _solid_cube_with_cavity(size=31, cavity_radius=4):
    bone = np.zeros((size, size, size), dtype=bool)
    bone[4:-4, 4:-4, 4:-4] = True
    zz, yy, xx = np.indices(bone.shape)
    center = np.array([(size - 1) / 2] * 3)
    cavity = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    ) <= cavity_radius**2
    bone[cavity] = False
    peri = np.zeros_like(bone)
    peri[4:-4, 4:-4, 4:-4] = True
    return bone, peri, cavity


def test_segment_voidspace_finds_large_internal_cavity():
    bone, peri, cavity = _solid_cube_with_cavity()
    params = VoidspaceParameters(
        closing_radius_mm=0.061,
        boundary_erosion_radius_mm=0.0,
        min_large_void_volume_mm3=0.001,
        bone_speckle_min_voxels=2,
        void_speckle_min_voxels=2,
    )

    result = segment_voidspace(bone, (0.061, 0.061, 0.061), mask=peri, parameters=params)

    assert result.all_void[cavity].any()
    assert result.large_void[cavity].any()
    assert not result.large_void[~peri].any()


def test_segment_voidspace_does_not_create_void_outside_periosteal_mask():
    bone = np.zeros((9, 9, 9), dtype=bool)
    peri = np.zeros_like(bone)
    peri[2:7, 2:7, 2:7] = True
    params = VoidspaceParameters(
        closing_radius_mm=0.122,
        boundary_erosion_radius_mm=0.0,
        min_large_void_volume_mm3=0.0,
        bone_speckle_min_voxels=1,
        void_speckle_min_voxels=1,
    )

    result = segment_voidspace(bone, (0.061, 0.061, 0.061), mask=peri, parameters=params)

    assert not result.all_void[~peri].any()


def test_segment_voidspace_derives_internal_domain_without_periosteal_mask():
    bone, peri, cavity = _solid_cube_with_cavity(size=21, cavity_radius=2)
    params = VoidspaceParameters(
        closing_radius_mm=0.061,
        boundary_erosion_radius_mm=0.0,
        min_large_void_volume_mm3=0.001,
        bone_speckle_min_voxels=1,
        void_speckle_min_voxels=1,
    )

    result = segment_voidspace(bone, (0.061, 0.061, 0.061), parameters=params)

    assert result.metadata["domain_source"] == "segmentation_border_background"
    assert result.large_void[cavity].any()
    assert not result.all_void[~peri].any()
