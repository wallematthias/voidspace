from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from voidspace.models import VoidspaceParameters
from voidspace.workflows import compare, run_case


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


def _cmd_compare(args: argparse.Namespace) -> int:
    compare(
        baseline_void_path=args.baseline_void,
        followup_void_path=args.followup_void,
        mask_path=args.mask,
        output_dir=args.output_dir,
        force=args.force,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
