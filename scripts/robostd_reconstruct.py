"""
RoboSTD: zero-shot Single-to-Dual demonstration construction (Algorithm 1).

Stage 0 (prerequisite): the sagittal-mirrored contralateral parquet, produced by
    python scripts/main.py --input <orig> --output <mir> --rule agilex --ops mirror
Stage 2 (this tool): LLM-guided spatio-temporal reconstruction that composes the
original single-arm parquet together with its mirrored version into a
pseudo-bimanual parquet (14-DoF layout ``[left(7), right(7)]``).

Usage
-----
    python scripts/robostd_reconstruct.py \
        --input data/RoboTwin/data/episode_000000.parquet \
        --mirror data/RoboTwin_mir/data/episode_000000.parquet \
        --output data/RoboTwin_bi/data/episode_000000.parquet \
        --source-arm R --n-units 4 \
        --language "place the bowl on the plate" \
        --planner openai            # or 'default' (deterministic, no API key)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.robostd_stage2 import (  # noqa: E402
    build_planner,
    decompose_units,
    parse_constraints_dict,
    rearrange_pseudo_bimanual,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("robostd_reconstruct")

STATE_COL = "observation.state"
ACTION_COL = "action"


def _read_14(df: pd.DataFrame, col: str) -> np.ndarray:
    vals = df[col].values
    first = vals[0]
    if isinstance(first, (list, np.ndarray)) and np.ndim(first) == 1:
        return np.vstack(vals).astype(np.float64)
    raise ValueError(f"Column {col} is not stored as 14-D vector sequences")


def _write_14(df_out: pd.DataFrame, col: str, arr: np.ndarray) -> None:
    df_out[col] = [np.asarray(row, dtype=np.float32) for row in arr]


def main():
    parser = argparse.ArgumentParser(description="RoboSTD Stage 2: pseudo-bimanual reconstruction")
    parser.add_argument("--input", required=True, help="Original single-arm parquet")
    parser.add_argument("--mirror", required=True, help="Sagittal-mirrored parquet (Stage 1 output)")
    parser.add_argument("--output", required=True, help="Output pseudo-bimanual parquet")
    parser.add_argument("--source-arm", default="R", choices=["L", "R"],
                        help="Arm active in the original single-arm demo")
    parser.add_argument("--n-units", type=int, default=None, help="Number of manipulation units")
    parser.add_argument("--unit-names", default=None,
                        help="Comma-separated unit names (e.g. 'grasp,lift,approach,place')")
    parser.add_argument("--planner", default="default", choices=["default", "openai"],
                        help="Coordination-constraint planner")
    parser.add_argument("--model", default="gpt-4.1", help="LLM model (openai planner)")
    parser.add_argument("--language", default="", help="Task language instruction (LLM context)")
    parser.add_argument("--objects", default="", help="Object identities/locations (LLM context)")
    parser.add_argument("--workspace", default="", help="Workspace/reachability/safety (LLM context)")
    parser.add_argument("--constraints-json", default=None,
                        help="Precomputed G as JSON (skips planner); OTHERS ignored")
    args = parser.parse_args()

    input_path, mirror_path, output_path = (Path(args.input), Path(args.mirror), Path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_parquet(input_path)
    df_mir = pd.read_parquet(mirror_path)

    state_orig = _read_14(df_orig, STATE_COL)
    state_mir = _read_14(df_mir, STATE_COL)
    action_orig = _read_14(df_orig, ACTION_COL)
    action_mir = _read_14(df_mir, ACTION_COL)

    n_frames = state_orig.shape[0]
    for name, arr in (("mirror", state_mir),):
        if arr.shape[0] != n_frames:
            raise ValueError(f"{name} parquet has {arr.shape[0]} frames, expected {n_frames}")

    unit_names = [s.strip() for s in args.unit_names.split(",")] if args.unit_names else None
    units = decompose_units(n_frames, args.n_units, unit_names)
    logger.info("Decomposed trajectory of %d frames into %d units", n_frames, len(units))

    task_context = {
        "language": args.language,
        "objects": args.objects or None,
        "workspace": args.workspace or None,
    }

    if args.constraints_json:
        import json

        constraints = parse_constraints_dict(json.loads(args.constraints_json))
        logger.info("Using precomputed constraints: %s", constraints)
    else:
        planner = build_planner(name=args.planner, source_arm=args.source_arm, model=args.model)
        constraints = planner.generate_constraints(units, task_context, args.source_arm)

    bi_state, bi_action, plan = rearrange_pseudo_bimanual(
        state_orig, state_mir, action_orig, action_mir,
        constraints=constraints, units=units, source_arm=args.source_arm,
    )

    for p in plan:
        logger.info("unit %d %-16s frames=%-10s arm=%s source=%s",
                    p["unit"], p["name"], str(p["frames"]), p["arm"], p["skill_source"])

    # Build the output parquet: reuse the observation columns that are not
    # the 14-d state/action (images, timestamps, indices, ...) unchanged.
    df_out = df_orig.copy()
    _write_14(df_out, STATE_COL, bi_state)
    _write_14(df_out, ACTION_COL, bi_action)
    df_out.to_parquet(output_path, engine="pyarrow")
    logger.info("Wrote pseudo-bimanual parquet: %s", output_path)


if __name__ == "__main__":
    main()