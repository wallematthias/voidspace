from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_mask(path: Path | str) -> tuple[np.ndarray, tuple[float, float, float], sitk.Image]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(bool)
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return array, spacing_zyx, image


def write_mask_like(mask: np.ndarray, reference: sitk.Image, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
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

