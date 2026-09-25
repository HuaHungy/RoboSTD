#!/usr/bin/env python3
"""Reconstruct one RoboSTD pseudo-bimanual parquet with Algorithm 1."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_reconstruction import _read_vectors, reconstruct_dataframe  # noqa: E402
from src.robostd_stage2 import (  # noqa: E402
    build_planner,
    decompose_units,
    parse_constraints_dict,
    parse_units_payload,
    reconstruct_pseudo_bimanual,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("robostd_reconstruct")


def _json_value(value: str):
    path = Path(value)
    text = path.read_text(encoding="utf-8-sig") if path.exists() else value
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="RoboSTD Stage 2 reconstruction")
    parser.add_argument("--input", required=True, help="Original single-arm parquet")
    parser.add_argument("--mirror", required=True, help="Stage 1 mirrored parquet")
    parser.add_argument("--output", required=True, help="Output pseudo-bimanual parquet")
    parser.add_argument("--source-arm", default="R", choices=["L", "R"])
    parser.add_argument("--n-units", type=int, default=4)
    parser.add_argument("--unit-names", help="Comma-separated semantic unit names")
    parser.add_argument("--unit-boundaries", help="Comma-separated boundaries including 0 and T")
    parser.add_argument("--units-json", help="JSON file/list with id, start, end, and name")
    parser.add_argument("--planner", default="openai", choices=["default", "openai"])
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--allow-planner-fallback", action="store_true")
    parser.add_argument("--language", default="")
    parser.add_argument("--objects", default="")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--constraints-json", help="Inline JSON or a JSON file")
    parser.add_argument("--action-mode", default="position", choices=["position", "delta"])
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--plan-output", help="Optional JSON schedule output")
    args = parser.parse_args()

    original = pd.read_parquet(args.input)
    mirrored = pd.read_parquet(args.mirror)
    if len(original) != len(mirrored):
        raise ValueError("original and mirrored episodes must have equal frame counts")
    state_orig = _read_vectors(original, "observation.state")
    state_mir = _read_vectors(mirrored, "observation.state")
    action_orig = _read_vectors(original, "action")
    action_mir = _read_vectors(mirrored, "action")

    if args.units_json:
        units = parse_units_payload(_json_value(args.units_json), len(original))
    else:
        names = [value.strip() for value in args.unit_names.split(",")] if args.unit_names else None
        boundaries = (
            [int(value) for value in args.unit_boundaries.split(",")]
            if args.unit_boundaries
            else None
        )
        units = decompose_units(len(original), args.n_units, names, boundaries)

    context = {
        "language": args.language,
        "objects": args.objects or None,
        "workspace": args.workspace or None,
    }
    if args.constraints_json:
        constraints = parse_constraints_dict(
            _json_value(args.constraints_json), [unit.id for unit in units]
        )
    else:
        planner = build_planner(
            args.planner,
            source_arm=args.source_arm,
            model=args.model,
            allow_fallback=args.allow_planner_fallback,
        )
        constraints = planner.generate_constraints(units, context, args.source_arm)

    result = reconstruct_pseudo_bimanual(
        state_orig,
        state_mir,
        action_orig,
        action_mir,
        constraints,
        units,
        args.source_arm,
        args.action_mode,
    )
    output = reconstruct_dataframe(
        original,
        mirrored,
        result,
        args.language,
        fps=args.fps,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, engine="pyarrow", index=False)

    plan_path = (
        Path(args.plan_output)
        if args.plan_output
        else output_path.with_suffix(".plan.json")
    )
    plan_path.write_text(
        json.dumps(
            {
                "constraints": {
                    "rho": constraints.rho,
                    "e_pre": constraints.e_pre,
                    "e_conf": constraints.e_conf,
                },
                "plan": result.plan,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s (%d frames) and %s", output_path, len(output), plan_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
