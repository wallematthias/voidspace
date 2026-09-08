import csv

import numpy as np
import py_aimio
import pytest
import SimpleITK as sitk

from voidspace.cli import main


def _write_mask(path, array):
    image = sitk.GetImageFromArray(array.astype("uint8"))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def _write_aim(path, array):
    metadata = {
        "dimensions": (array.shape[2], array.shape[1], array.shape[0]),
        "element_size": (1.0, 1.0, 1.0),
        "spacing": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "position": (0, 0, 0),
        "offset": (0, 0, 0),
        "processing_log_raw": "Created for voidspace test",
    }
    py_aimio.write_aim(str(path), (127 * array).astype(np.int8), metadata, unit="native")


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
            "--mask",
            str(tmp_path / "common.nii.gz"),
            "--output-dir",
            str(tmp_path / "out"),
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
    with (tmp_path / "out" / "voidspace_measurements.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert "subject_id" not in rows[0]
    assert "analysis_masked" not in rows[0]


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
            "--force",
        ]
    )

    assert rc == 0
    assert (tmp_path / "change" / "voidspace_expanded_mask.nii.gz").is_file()
    assert (tmp_path / "change" / "voidspace_contracted_mask.nii.gz").is_file()


def test_run_case_cli_writes_aim_masks_for_aim_input(tmp_path):
    segmentation = np.ones((9, 9, 9), dtype=bool)
    segmentation[4, 4, 4] = False
    sample_aim = tmp_path / "seg.AIM"
    _write_aim(sample_aim, segmentation)

    rc = main(
        [
            "run-case",
            "--segmentation",
            str(sample_aim),
            "--output-dir",
            str(tmp_path / "aim-out"),
            "--min-large-void-volume-mm3",
            "0.0",
            "--boundary-erosion-radius-mm",
            "0.0",
            "--force",
        ]
    )

    assert rc == 0
    assert (tmp_path / "aim-out" / "voidspace_large_mask.AIM").is_file()
    assert (tmp_path / "aim-out" / "voidspace_all_mask.AIM").is_file()


def test_run_case_help_uses_single_mask_argument_and_no_study_metadata(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run-case", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--mask" in help_text
    assert "--analysis-mask" not in help_text
    assert "--periosteal-mask" not in help_text
    assert "--subject" not in help_text
    assert "--session" not in help_text
    assert "--site" not in help_text
