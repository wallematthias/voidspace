import numpy as np

from voidspace.morphology import (
    ellipsoid_footprint,
    min_voxels_for_volume,
    remove_small_components,
    voxel_volume_mm3,
)


def test_ellipsoid_footprint_is_symmetric_for_isotropic_spacing():
    footprint = ellipsoid_footprint(0.122, (0.061, 0.061, 0.061))

    assert footprint.shape == (5, 5, 5)
    assert footprint[2, 2, 2]
    assert np.array_equal(footprint, footprint[::-1, :, :])
    assert np.array_equal(footprint, footprint[:, ::-1, :])
    assert np.array_equal(footprint, footprint[:, :, ::-1])


def test_min_voxels_for_volume_uses_physical_spacing():
    assert voxel_volume_mm3((0.061, 0.061, 0.061)) == 0.061**3
    assert min_voxels_for_volume(16.5, (0.061, 0.061, 0.061)) == 72694


def test_remove_small_components_removes_only_components_below_threshold():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1, 1, 1] = True
    mask[3:5, 3:5, 3:5] = True

    cleaned = remove_small_components(mask, min_voxels=6, connectivity=3)

    assert not cleaned[1, 1, 1]
    assert cleaned[3:5, 3:5, 3:5].all()
