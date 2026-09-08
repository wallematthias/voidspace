import numpy as np

from voidspace import measure_voidspace


def test_measure_voidspace_reports_volume_ratio_and_count():
    void = np.zeros((5, 5, 5), dtype=bool)
    total = np.ones_like(void, dtype=bool)
    void[1, 1, 1] = True
    void[3, 3, 3] = True

    metrics = measure_voidspace(void, total, (1.0, 1.0, 1.0))

    assert metrics.volume_mm3 == 2.0
    assert metrics.total_volume_mm3 == 125.0
    assert metrics.vstv_percent == 1.6
    assert metrics.component_count == 2


def test_measure_voidspace_applies_mask_to_measurement_domain():
    void = np.zeros((4, 4, 4), dtype=bool)
    total = np.ones_like(void, dtype=bool)
    common = np.zeros_like(void, dtype=bool)
    common[:2, :, :] = True
    void[0, 0, 0] = True
    void[3, 3, 3] = True

    metrics = measure_voidspace(void, total, (1.0, 1.0, 1.0), mask=common)

    assert metrics.volume_mm3 == 1.0
    assert metrics.total_volume_mm3 == 32.0
    assert metrics.vstv_percent == 3.125
