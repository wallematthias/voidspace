from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class AimReference:
    path: Path
    metadata: dict[str, Any]
    image: sitk.Image


ImageReference = sitk.Image | AimReference


def is_aim_path(path: Path | str) -> bool:
    return re.search(r"\.aim(?:;\d+)?$", Path(path).name, re.IGNORECASE) is not None


def _load_py_aimio():
    try:
        import py_aimio
    except ImportError as exc:
        raise RuntimeError(
            "AIM files require the PyPI package 'aimio-py'. Install with "
            "`python -m pip install aimio-py`."
        ) from exc
    return py_aimio


def _spacing_zyx_from_image(image: sitk.Image) -> tuple[float, float, float]:
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    return (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])


def _as_zyx(array: np.ndarray, dimensions_xyz: tuple[int, int, int] | None) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D AIM array, got shape {array.shape}.")
    if dimensions_xyz is None:
        return array
    expected_zyx = (dimensions_xyz[2], dimensions_xyz[1], dimensions_xyz[0])
    if tuple(array.shape) == expected_zyx:
        return array
    if tuple(array.shape) == dimensions_xyz:
        return np.transpose(array, (2, 1, 0))
    return array


def _image_from_aim_array(array: np.ndarray, metadata: dict[str, Any]) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    spacing = metadata.get("element_size", metadata.get("spacing", (1.0, 1.0, 1.0)))
    if not (isinstance(spacing, (tuple, list)) and len(spacing) == 3):
        spacing = (1.0, 1.0, 1.0)
    image.SetSpacing(tuple(float(v) for v in spacing))
    origin = metadata.get("origin", (0.0, 0.0, 0.0))
    if isinstance(origin, (tuple, list)) and len(origin) >= 3:
        image.SetOrigin(tuple(float(v) for v in origin[:3]))
    direction = metadata.get("direction")
    if isinstance(direction, (tuple, list)) and len(direction) == 9:
        image.SetDirection(tuple(float(v) for v in direction))
    return image


def _read_aim_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float], AimReference]:
    py_aimio = _load_py_aimio()
    array, metadata = py_aimio.read_aim(str(path), density=False, hu=False)
    metadata = dict(metadata)
    dimensions_raw = metadata.get("dimensions")
    dimensions_xyz = (
        tuple(int(v) for v in dimensions_raw)
        if isinstance(dimensions_raw, (tuple, list)) and len(dimensions_raw) == 3
        else None
    )
    array = _as_zyx(np.asarray(array), dimensions_xyz)
    image = _image_from_aim_array(array, metadata)
    return array.astype(bool), _spacing_zyx_from_image(image), AimReference(path, metadata, image)


def read_mask(path: Path | str) -> tuple[np.ndarray, tuple[float, float, float], ImageReference]:
    path = Path(path)
    if is_aim_path(path):
        return _read_aim_mask(path)

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(bool)
    return array, _spacing_zyx_from_image(image), image


def _write_aim_mask(mask: np.ndarray, reference: AimReference, path: Path) -> None:
    py_aimio = _load_py_aimio()
    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    image.CopyInformation(reference.image)
    metadata = dict(reference.metadata)
    metadata["dimensions"] = tuple(int(v) for v in image.GetSize())
    metadata["spacing"] = tuple(float(v) for v in image.GetSpacing())
    metadata["element_size"] = tuple(float(v) for v in image.GetSpacing())
    metadata["origin"] = tuple(float(v) for v in image.GetOrigin())
    metadata["direction"] = tuple(float(v) for v in image.GetDirection())
    metadata["unit"] = "native"
    if isinstance(metadata.get("processing_log"), dict) and "processing_log_raw" not in metadata:
        metadata["processing_log_raw"] = py_aimio.dict_to_log(metadata["processing_log"])
    array = (127 * (sitk.GetArrayFromImage(image) > 0)).astype(np.int8)
    py_aimio.write_aim(str(path), array, metadata, unit="native")


def write_mask_like(mask: np.ndarray, reference: ImageReference, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(reference, AimReference) or is_aim_path(output):
        if not isinstance(reference, AimReference):
            raise ValueError("AIM output requires an AIM reference image.")
        _write_aim_mask(mask, reference, output)
        return output

    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(output))
    return output


def write_metrics_csv(path: Path | str, rows: list[dict[str, object]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output
