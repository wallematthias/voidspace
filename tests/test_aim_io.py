import numpy as np
import py_aimio

from voidspace.io import is_aim_path, read_mask, write_mask_like


def _write_aim(path, array, spacing=(0.082, 0.082, 0.082)):
    metadata = {
        "dimensions": (array.shape[2], array.shape[1], array.shape[0]),
        "element_size": spacing,
        "spacing": spacing,
        "origin": (0.0, 0.0, 0.0),
        "position": (0, 0, 0),
        "offset": (0, 0, 0),
        "processing_log_raw": "Created for voidspace test",
    }
    py_aimio.write_aim(str(path), array.astype(np.int8), metadata, unit="native")


def test_read_mask_reads_scanco_aim(tmp_path):
    source = np.zeros((3, 4, 5), dtype=np.int8)
    source[1, 2, 3] = 127
    path = tmp_path / "seg.AIM"
    _write_aim(path, source)

    segmentation, spacing, reference = read_mask(path)

    assert is_aim_path(path)
    assert segmentation.shape == source.shape
    assert segmentation.dtype == bool
    assert int(segmentation.sum()) == 1
    assert spacing == (
        0.0820000022649765,
        0.0820000022649765,
        0.0820000022649765,
    )
    assert reference.path == path


def test_write_mask_like_round_trips_scanco_aim(tmp_path):
    source = np.zeros((3, 4, 5), dtype=np.int8)
    source[1, 2, 3] = 127
    input_path = tmp_path / "seg.AIM"
    _write_aim(input_path, source)
    segmentation, _spacing, reference = read_mask(input_path)
    subset = segmentation.copy()

    output = write_mask_like(subset, reference, tmp_path / "voidspace_large_mask.AIM")
    roundtrip, roundtrip_spacing, _roundtrip_reference = read_mask(output)

    assert output.is_file()
    assert roundtrip.shape == segmentation.shape
    assert roundtrip_spacing == (
        0.0820000022649765,
        0.0820000022649765,
        0.0820000022649765,
    )
    assert np.array_equal(roundtrip, subset)
