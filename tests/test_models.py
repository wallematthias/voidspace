from voidspace import VoidspaceParameters


def test_xtremectii_defaults_are_physical_algorithm_values():
    params = VoidspaceParameters.xtremectii_defaults()

    assert params.closing_radius_mm == 0.738
    assert params.boundary_erosion_radius_mm == 0.366
    assert params.min_large_void_volume_mm3 == 16.5
    assert params.bone_speckle_min_voxels == 6
    assert params.void_speckle_min_voxels == 6


def test_parameters_reject_negative_values():
    try:
        VoidspaceParameters(closing_radius_mm=-0.1)
    except ValueError as exc:
        assert "closing_radius_mm must be >= 0" in str(exc)
    else:
        raise AssertionError("negative closing radius was accepted")

