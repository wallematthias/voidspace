# Changelog

## 0.1.4 - 2026-10-02

- Preserve large cavities connected to acquisition ends by excluding lateral background only when inferring a domain.
- Use endpoint-continuation padding for morphology to reduce artificial scan-end erosion/dilation effects; prefer supplied full bone masks.
- Reconcile cropped and displaced masks using the complete physical reference grid and nearest-neighbor resampling for native, registered, and dynamic workflows.
