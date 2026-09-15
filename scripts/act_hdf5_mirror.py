"""
python '/home/huahungy/act/scripts/RoboSTD/scripts/act_hdf5_mirror.py' \
    --input_dir act_dataset/Agilex_Cobot_Magic_Put_the_towel_in_the_basket_left_0508/ \
    --output_dir act_dataset/Agilex_Cobot_Magic_Put_the_towel_in_the_basket_left_0508_mirrored \
    --joint_mode aloha14
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.act_hdf5_mirror import mirror_act_episode_hdf5


def _episode_key(p: Path):
    m = re.match(r"episode_(\d+)\.hdf5$", p.name)
    if m:
        return int(m.group(1))
    return p.name


def main():
    parser = argparse.ArgumentParser(description="Mirror ACT-style per-episode HDF5 dataset")
    parser.add_argument("--input_dir", required=True, help="Directory containing episode_*.hdf5")
    parser.add_argument("--output_dir", required=True, help="Directory to write mirrored episode_*.hdf5")
    parser.add_argument("--joint_mode", default="aloha14", choices=["aloha14"])
    parser.add_argument("--no_swap_cameras", action="store_true")
    parser.add_argument("--frame_batch", type=int, default=16)
    parser.add_argument("--episodes", default=None, help="Comma separated episode indices (e.g. 0,1,2)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N episodes after filtering")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(input_dir.glob("episode_*.hdf5"), key=_episode_key)
    if not inputs:
        raise RuntimeError(f"No episode_*.hdf5 found in {input_dir}")

    if args.episodes:
        allowed = set(int(x.strip()) for x in args.episodes.split(",") if x.strip())
        inputs = [p for p in inputs if _episode_key(p) in allowed]
        if not inputs:
            raise RuntimeError("No episodes matched --episodes")

    if args.limit is not None:
        inputs = inputs[: max(0, int(args.limit))]

    for i, in_path in enumerate(inputs, start=1):
        print(f"[{i}/{len(inputs)}] {in_path.name}", flush=True)
        out_path = output_dir / in_path.name
        if out_path.exists() and args.skip_existing:
            continue
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_path} (use --overwrite to replace)")
        mirror_act_episode_hdf5(
            input_h5=in_path,
            output_h5=out_path,
            joint_mode=args.joint_mode,
            swap_cameras=not args.no_swap_cameras,
            frame_batch=args.frame_batch,
        )


if __name__ == "__main__":
    main()
