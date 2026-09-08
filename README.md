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
- Masked/common-region: pass a segmented bone image and an analysis mask. The
  full voidspace mask is computed first, then the mask is applied to the
  measurements.
- Registered: pass an already registered segmentation and optional registered
  analysis mask.
- Dynamic change: pass already aligned baseline and follow-up voidspace masks.

When no periosteal mask is supplied, the segmentation-only workflow estimates
the analysis domain from the closed segmentation by excluding background
connected to the image border. Supplying an explicit periosteal/common-region
mask remains preferred when that contour is available.

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
  --analysis-mask common_region.nii.gz \
  --output-dir derivatives/VoidSpace/sub-S1/site-tibia/ses-1 \
  --subject S1 \
  --session 1 \
  --site tibia
```

```bash
voidspace compare \
  --baseline-void baseline_voidspace_large_mask.nii.gz \
  --followup-void followup_voidspace_large_mask.nii.gz \
  --analysis-mask registered_common_region.nii.gz \
  --output-dir derivatives/VoidSpace/sub-S1/site-tibia/t0-1_t1-2 \
  --subject S1 \
  --site tibia \
  --baseline-session 1 \
  --followup-session 2
```
