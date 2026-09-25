#!/usr/bin/env python3
"""Derive Stage 1 joint signs from a symmetric robot URDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.geometry_mirror import (  # noqa: E402
    build_bidirectional_rules,
    derive_joint_signs_from_urdf,
)


def _csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive RoboSTD sagittal joint signs from common-frame URDF axes"
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--left-joints", required=True, help="Comma-separated URDF joint names")
    parser.add_argument("--right-joints", required=True, help="Comma-separated URDF joint names")
    parser.add_argument(
        "--left-fields", help="Comma-separated dataset field names; defaults to joint names"
    )
    parser.add_argument(
        "--right-fields", help="Comma-separated dataset field names; defaults to joint names"
    )
    parser.add_argument(
        "--plane-normal",
        default="0,1,0",
        help="Sagittal-plane normal in the URDF root frame (default: 0,1,0)",
    )
    parser.add_argument("--rule-name", default="urdf_derived")
    parser.add_argument("--output", required=True, help="Output YAML file")
    args = parser.parse_args()

    left_joints = _csv(args.left_joints)
    right_joints = _csv(args.right_joints)
    left_fields = _csv(args.left_fields) if args.left_fields else left_joints
    right_fields = _csv(args.right_fields) if args.right_fields else right_joints
    normal = [float(value) for value in _csv(args.plane_normal)]

    signs = derive_joint_signs_from_urdf(
        args.urdf,
        left_joints,
        right_joints,
        normal=normal,
    )
    rules = build_bidirectional_rules(left_fields, right_fields, signs)
    payload = {
        "geometry": {
            "urdf": str(Path(args.urdf)),
            "sagittal_plane_normal": normal,
            "left_joints": left_joints,
            "right_joints": right_joints,
            "lambda": signs,
        },
        "joint_mirror_rules": {args.rule_name: rules},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    print(f"Wrote {output} with Lambda={signs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
