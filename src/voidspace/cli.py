from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from voidspace.models import VoidspaceParameters
from voidspace.workflows import analyze_maps, compare, intersect_masks, run_case


def _parameters_from_args(args: argparse.Namespace) -> VoidspaceParameters:
    return VoidspaceParameters(
        closing_radius_mm=args.closing_radius_mm,
        boundary_erosion_radius_mm=args.boundary_erosion_radius_mm,
        min_large_void_volume_mm3=args.min_large_void_volume_mm3,
        bone_speckle_min_voxels=args.bone_speckle_min_voxels,
        void_speckle_min_voxels=args.void_speckle_min_voxels,
        connectivity=args.connectivity,
    )


def _add_parameter_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = VoidspaceParameters.xtremectii_defaults()
    parser.add_argument("--closing-radius-mm", type=float, default=defaults.closing_radius_mm)
    parser.add_argument(
        "--boundary-erosion-radius-mm",
        type=float,
        default=defaults.boundary_erosion_radius_mm,
    )
    parser.add_argument(
        "--min-large-void-volume-mm3",
        type=float,
        default=defaults.min_large_void_volume_mm3,
    )
    parser.add_argument(
        "--bone-speckle-min-voxels",
        type=int,
        default=defaults.bone_speckle_min_voxels,
    )
    parser.add_argument(
        "--void-speckle-min-voxels",
        type=int,
        default=defaults.void_speckle_min_voxels,
    )
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=defaults.connectivity)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voidspace",
        description="Run spacing-aware voidspace analysis on prepared HR-pQCT masks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_case = subparsers.add_parser("run-case", help="Segment and measure one prepared scan.")
    run_case.add_argument("--segmentation", type=Path, required=True)
    run_case.add_argument(
        "--mask",
        type=Path,
        help="Optional analysis-domain mask. Combine periosteal/common-region masks before passing.",
    )
    run_case.add_argument("--output-dir", type=Path, required=True)
    run_case.add_argument("--force", action="store_true")
    _add_parameter_arguments(run_case)
    run_case.set_defaults(func=_cmd_run_case)

    analyze = subparsers.add_parser("analyze-maps", help="Measure existing voidspace maps inside a mask.")
    analyze.add_argument("--large-void", type=Path, required=True)
    analyze.add_argument("--all-void", type=Path, required=True)
    analyze.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Analysis-domain mask aligned with both voidspace maps.",
    )
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=VoidspaceParameters.xtremectii_defaults().connectivity)
    analyze.add_argument("--force", action="store_true")
    analyze.set_defaults(func=_cmd_analyze_maps)

    compare = subparsers.add_parser("compare", help="Compare already aligned voidspace masks.")
    compare.add_argument("--baseline-void", type=Path, required=True)
    compare.add_argument("--followup-void", type=Path, required=True)
    compare.add_argument(
        "--mask",
        type=Path,
        help="Optional analysis-domain mask aligned with both voidspace masks.",
    )
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--force", action="store_true")
    compare.set_defaults(func=_cmd_compare)

    intersect = subparsers.add_parser("intersect-masks", help="Write the intersection of aligned masks.")
    intersect.add_argument(
        "--mask",
        type=Path,
        required=True,
        action="append",
        help="Mask to include in the intersection. Pass at least two.",
    )
    intersect.add_argument("--output", type=Path, required=True)
    intersect.add_argument("--force", action="store_true")
    intersect.set_defaults(func=_cmd_intersect_masks)

    return parser


def _cmd_run_case(args: argparse.Namespace) -> int:
    run_case(
        segmentation_path=args.segmentation,
        mask_path=args.mask,
        output_dir=args.output_dir,
        parameters=_parameters_from_args(args),
        force=args.force,
    )
    return 0


def _cmd_analyze_maps(args: argparse.Namespace) -> int:
    analyze_maps(
        large_void_path=args.large_void,
        all_void_path=args.all_void,
        mask_path=args.mask,
        output_dir=args.output_dir,
        connectivity=args.connectivity,
        force=args.force,
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    compare(
        baseline_void_path=args.baseline_void,
        followup_void_path=args.followup_void,
        mask_path=args.mask,
        output_dir=args.output_dir,
        force=args.force,
    )
    return 0


def _cmd_intersect_masks(args: argparse.Namespace) -> int:
    intersect_masks(args.mask, output_path=args.output, force=args.force)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
