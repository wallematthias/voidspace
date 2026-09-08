# voidspace

Spacing-aware voidspace analysis for HR-pQCT segmentations.

`voidspace` is a standalone logic package. It does not segment bone, register
scans, create common regions, or resample images. It expects inputs that are
already in the image space to analyze.

The command line interface reads common SimpleITK image formats and Scanco AIM
masks. AIM masks are read in native scaling, interpreted as foreground where
nonzero, and written back as native binary AIM masks using the input geometry.

## Workflows

- Cross-sectional: pass a segmented bone image.
- Masked/common-region: pass a segmented bone image and a mask. If both a
  periosteal contour and a common region are needed, combine them before calling
  `voidspace`.
- Registered: pass an already registered segmentation and optional registered
  mask.
- Dynamic change: pass already aligned baseline and follow-up voidspace masks.

When no periosteal mask is supplied, the segmentation-only workflow estimates
the analysis domain from the closed segmentation by excluding background
connected to the image border. Supplying an explicit mask remains preferred
when that contour is available.

## Algorithm

The algorithm follows the original IPL voidspace concept:

1. Remove tiny disconnected bone components.
2. Close segmented bone to fill normal trabecular spacing.
3. Invert the closed bone region to candidate marrow void.
4. Erode candidate void away from trabecular surfaces.
5. Remove tiny candidate void components.
6. Keep all surviving voids and large voids above a physical volume threshold.

Unlike the IPL scripts, morphology uses a spacing-aware spherical or ellipsoidal
footprint. The default XtremeCT II settings are:

- closing radius: `0.738 mm`
- boundary erosion radius: `0.366 mm`
- minimum large-void volume: `16.5 mm3`
- bone and void speckle cleanup: `6 voxels`

The original IPL code used `72560` or `72660` voxels as scanner-specific large
void thresholds in different script/paper contexts. This package treats
`16.5 mm3` as the default source of truth and converts it from the supplied
image spacing.

## CLI

```bash
voidspace run-case \
  --segmentation seg.AIM \
  --mask analysis_domain.AIM \
  --output-dir voidspace-out
```

```bash
voidspace compare \
  --baseline-void baseline_voidspace_large_mask.AIM \
  --followup-void followup_voidspace_large_mask.AIM \
  --mask analysis_domain.AIM \
  --output-dir voidspace-change
```

`--mask` is optional for both commands. Use `--force` to overwrite existing
outputs.

## Python

```python
from voidspace import VoidspaceParameters, run_case

result = run_case(
    segmentation_path="seg.AIM",
    mask_path="analysis_domain.AIM",
    output_dir="voidspace-out",
    parameters=VoidspaceParameters.xtremectii_defaults(),
)
print(result.metrics.volume_mm3)
```

```python
from voidspace import compare

result = compare(
    baseline_void_path="baseline_voidspace_large_mask.AIM",
    followup_void_path="followup_voidspace_large_mask.AIM",
    mask_path="analysis_domain.AIM",
    output_dir="voidspace-change",
)
print(result.metrics.net_change_volume_mm3)
```
