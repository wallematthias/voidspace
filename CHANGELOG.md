# Changelog

## 0.1.6 - 2026-10-05

- Synchronize the public `voidspace.__version__` with the package release version. Includes all spacing/alignment repairs from 0.1.5 without further algorithm changes.

## 0.1.5 - 2026-10-05

- Accept AIM/NIfTI mask-spacing header rounding up to 0.000001 mm per axis when intersecting full masks with common regions, preserving physical placement and the reference grid.
- Reject genuinely different mask resolutions with an error listing both paths and spacings.
- Add regression tests for scanner header rounding and cropped-mask placement.

## 0.1.4 - 2026-10-02

- Preserve large cavities connected to acquisition ends by excluding lateral background only when inferring a domain.
- Use endpoint-continuation padding for morphology to reduce artificial scan-end erosion/dilation effects; prefer supplied full bone masks.
- Reconcile cropped and displaced masks using the complete physical reference grid and nearest-neighbor resampling for native, registered, and dynamic workflows.
