import numpy as np

from voidspace import classify_voidspace_change, measure_voidspace_change


def test_classify_voidspace_change_reports_stable_expanded_contracted():
    baseline = np.zeros((3, 3, 3), dtype=bool)
    followup = np.zeros_like(baseline)
    baseline[1, 1, 1] = True
    baseline[0, 0, 0] = True
    followup[1, 1, 1] = True
    followup[2, 2, 2] = True

    change = classify_voidspace_change(baseline, followup)

    assert change.stable.sum() == 1
    assert change.contracted.sum() == 1
    assert change.expanded.sum() == 1


def test_measure_voidspace_change_reports_net_volume():
    change = classify_voidspace_change(
        np.array([[[True, True, False]]]),
        np.array([[[False, True, True]]]),
    )

    metrics = measure_voidspace_change(change, (2.0, 1.0, 1.0))

    assert metrics.stable_volume_mm3 == 2.0
    assert metrics.contracted_volume_mm3 == 2.0
    assert metrics.expanded_volume_mm3 == 2.0
    assert metrics.net_change_volume_mm3 == 0.0

