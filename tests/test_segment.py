import numpy as np

from voidspace import VoidspaceParameters, segment_voidspace


def _open_tube():
    yy, xx = np.indices((65, 65))
    radius = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
    full = np.broadcast_to(radius < 25, (31, 65, 65)).copy()
    cavity = np.broadcast_to(radius < 15, full.shape)
    params = VoidspaceParameters(closing_radius_mm=1, boundary_erosion_radius_mm=1,
                                min_large_void_volume_mm3=10,
                                bone_speckle_min_voxels=1, void_speckle_min_voxels=1)
    return full & ~cavity, full, params


def test_open_end_void_survives_without_explicit_domain():
    bone, full, params = _open_tube()
    result = segment_voidspace(bone, (1, 1, 1), parameters=params)
    assert result.large_void[:, 32, 32].all()
    assert not result.all_void[~full].any()
    assert not np.any(result.large_void & ~result.all_void)


def test_end_slices_match_interior_for_uniform_tube():
    bone, full, params = _open_tube()
    result = segment_voidspace(bone, (1, 1, 1), mask=full, parameters=params)
    assert np.array_equal(result.filled_bone[0], result.filled_bone[15])
    assert np.array_equal(result.all_void[0], result.all_void[15])
    assert np.array_equal(result.all_void[-1], result.all_void[15])


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

    assert result.metadata["domain_source"] == "segmentation_lateral_background"
    assert result.large_void[cavity].any()
    assert not result.all_void[~peri].any()
