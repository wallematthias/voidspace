import csv

import numpy as np
import SimpleITK as sitk

from voidspace.cli import main


def _write_mask(path, array):
    image = sitk.GetImageFromArray(array.astype("uint8"))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def test_run_case_cli_writes_masks_and_measurements(tmp_path):
    segmentation = np.ones((9, 9, 9), dtype=bool)
    segmentation[4, 4, 4] = False
    common = np.zeros_like(segmentation)
    common[:5, :, :] = True
    _write_mask(tmp_path / "seg.nii.gz", segmentation)
    _write_mask(tmp_path / "common.nii.gz", common)

    rc = main(
        [
            "run-case",
            "--segmentation",
            str(tmp_path / "seg.nii.gz"),
            "--analysis-mask",
            str(tmp_path / "common.nii.gz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--subject",
            "S1",
            "--session",
            "1",
            "--site",
            "tibia",
            "--min-large-void-volume-mm3",
            "0.0",
            "--boundary-erosion-radius-mm",
            "0.0",
            "--force",
        ]
    )

    assert rc == 0
    assert (tmp_path / "out" / "voidspace_large_mask.nii.gz").is_file()
    assert (tmp_path / "out" / "voidspace_measurements.csv").is_file()
    assert (tmp_path / "out" / "voidspace_analysis_masked_measurements.csv").is_file()
    with (tmp_path / "out" / "voidspace_analysis_masked_measurements.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["analysis_masked"] == "True"


def test_compare_cli_writes_expanded_and_contracted_masks(tmp_path):
    baseline = np.zeros((2, 2, 2), dtype=bool)
    followup = np.zeros_like(baseline)
    baseline[0, 0, 0] = True
    followup[1, 1, 1] = True
    _write_mask(tmp_path / "baseline.nii.gz", baseline)
    _write_mask(tmp_path / "followup.nii.gz", followup)

    rc = main(
        [
            "compare",
            "--baseline-void",
            str(tmp_path / "baseline.nii.gz"),
            "--followup-void",
            str(tmp_path / "followup.nii.gz"),
            "--output-dir",
            str(tmp_path / "change"),
            "--subject",
            "S1",
            "--site",
            "tibia",
            "--baseline-session",
            "1",
            "--followup-session",
            "2",
            "--force",
        ]
    )

    assert rc == 0
    assert (tmp_path / "change" / "voidspace_expanded_mask.nii.gz").is_file()
    assert (tmp_path / "change" / "voidspace_contracted_mask.nii.gz").is_file()
