#!/usr/bin/env python3
"""
python scripts/lerobot_image_mirror.py \
  --input-root ./datasets/lerobot_input \
  --mode full \
  --joint-rule-config ./configs/agilex_lerobot_mirror.yaml \
  --rule-name agilex \
  --overwrite

Mirror a LeRobot image-path dataset into a sibling output directory.

This script is designed for datasets whose visual data lives under:
  images/{image_key}/episode_xxxxxx/frame_xxxxxx.jpg

Current modes:
  - image_only: mirror images and copy non-image assets as-is
  - full: mirror images and transform parquet joint columns with YAML rules

Rule config format for full mode:

joint_mirror_rules:
  agilex:
    - source: leader_joint1_right.pos
      target: leader_joint1_left.pos
      scale: 1.0
    - source: leader_joint1_left.pos
      target: leader_joint1_right.pos
      scale: 1.0
    - source: follower_joint1_right.pos
      target: follower_joint1_left.pos
      scale: 1.0
    - source: follower_joint1_left.pos
      target: follower_joint1_right.pos
      scale: 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


LOGGER = logging.getLogger("lerobot_image_mirror")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror a LeRobot image-path dataset into a sibling directory."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory of the input LeRobot dataset.",
    )
    parser.add_argument(
        "--output-root",
        help="Output directory. Defaults to <input-root>_mirrored beside the input.",
    )
    parser.add_argument(
        "--mode",
        choices=("image_only", "full"),
        default="image_only",
        help="Whether to mirror images only or images + parquet joint data.",
    )
    parser.add_argument(
        "--joint-rule-config",
        help="YAML file that defines joint mirror rules for full mode.",
    )
    parser.add_argument(
        "--rule-name",
        help="Rule set name inside joint_mirror_rules when the YAML contains multiple rule groups.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_default_output_root(input_root: Path) -> Path:
    return input_root.parent / f"{input_root.name}_mirrored"


def load_dataset_info(input_root: Path) -> dict:
    info_path = input_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    if not info.get("image_path"):
        raise ValueError(
            "This dataset does not declare image_path in meta/info.json. "
            "The current script only supports image-path LeRobot datasets."
        )

    return info


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def swap_left_right_token(name: str) -> str:
    replacements = [
        ("left", "__TMP_LEFT_RIGHT__"),
        ("right", "left"),
        ("__TMP_LEFT_RIGHT__", "right"),
        ("Left", "__TMP_LEFT_RIGHT__"),
        ("Right", "Left"),
        ("__TMP_LEFT_RIGHT__", "Right"),
        ("LEFT", "__TMP_LEFT_RIGHT__"),
        ("RIGHT", "LEFT"),
        ("__TMP_LEFT_RIGHT__", "RIGHT"),
    ]

    swapped = name
    for old, new in replacements:
        swapped = swapped.replace(old, new)
    return swapped


def get_image_keys(info: dict) -> List[str]:
    image_keys = []
    for feature_name, feature_info in info.get("features", {}).items():
        if feature_info.get("dtype") == "image":
            image_keys.append(feature_name)
    if not image_keys:
        raise ValueError("No image features were found in meta/info.json.")
    return sorted(image_keys)


def build_image_plan(image_keys: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    paired_keys: List[Tuple[str, str]] = []
    independent_keys: List[str] = []
    seen = set()
    image_key_set = set(image_keys)

    for key in image_keys:
        if key in seen:
            continue

        swapped = swap_left_right_token(key)
        if swapped != key and swapped in image_key_set:
            paired_keys.append((key, swapped))
            seen.add(key)
            seen.add(swapped)
        else:
            independent_keys.append(key)
            seen.add(key)

    return paired_keys, independent_keys


def copy_non_image_assets(input_root: Path, output_root: Path, skip_data: bool) -> None:
    for item in input_root.iterdir():
        if item.name == "images":
            continue
        if skip_data and item.name == "data":
            continue

        destination = output_root / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def mirror_image_file(src_path: Path, dst_path: Path) -> None:
    image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {src_path}")

    mirrored = cv2.flip(image, 1)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst_path), mirrored):
        raise ValueError(f"Failed to write image: {dst_path}")


def mirror_tree(src_root: Path, dst_root: Path) -> int:
    if not src_root.exists():
        LOGGER.warning("Image directory does not exist, skipping: %s", src_root)
        return 0

    processed = 0
    for src_file in iter_files(src_root):
        relative = src_file.relative_to(src_root)
        dst_file = dst_root / relative

        if src_file.suffix.lower() in IMAGE_SUFFIXES:
            mirror_image_file(src_file, dst_file)
            processed += 1
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    return processed


def build_field_map(info: dict) -> Dict[str, List[Tuple[str, int]]]:
    field_map: Dict[str, List[Tuple[str, int]]] = {}
    for feature_name, feature_info in info.get("features", {}).items():
        names = feature_info.get("names")
        if not names:
            continue

        if isinstance(names, list) and names and isinstance(names[0], list):
            names = names[0]

        if not isinstance(names, list):
            continue

        for idx, name in enumerate(names):
            if not isinstance(name, str):
                continue
            field_map.setdefault(name, []).append((feature_name, idx))

    return field_map


def load_joint_rules(config_path: Path, rule_name: str | None) -> List[dict]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    rules = config.get("joint_mirror_rules")
    if rules is None:
        raise ValueError(
            f"Missing 'joint_mirror_rules' in rule config: {config_path}"
        )

    if isinstance(rules, list):
        return rules

    if not isinstance(rules, dict):
        raise ValueError(
            "'joint_mirror_rules' must be either a list or a dict of named rule sets."
        )

    if rule_name:
        if rule_name not in rules:
            raise ValueError(
                f"Rule name '{rule_name}' was not found in {config_path}."
            )
        selected = rules[rule_name]
    elif len(rules) == 1:
        selected = next(iter(rules.values()))
    else:
        raise ValueError(
            "The rule config contains multiple named rule sets. "
            "Please provide --rule-name."
        )

    if not isinstance(selected, list):
        raise ValueError("Selected joint mirror rule set must be a list.")

    return selected


def read_parquet_table(parquet_path: Path) -> pa.Table:
    parquet_file = pq.ParquetFile(parquet_path)
    return parquet_file.read()


def transform_table_with_rules(
    table: pa.Table,
    field_map: Dict[str, List[Tuple[str, int]]],
    rules: Sequence[dict],
) -> pa.Table:
    involved_columns = set()
    validated_rules = []

    for rule in rules:
        source = rule.get("source")
        target = rule.get("target")
        scale = float(rule.get("scale", 1.0))

        if not source or not target:
            raise ValueError(f"Invalid rule entry: {rule}")

        source_locs = field_map.get(source, [])
        target_locs = field_map.get(target, [])
        if not source_locs:
            raise ValueError(f"Source field '{source}' was not found in dataset info.")
        if not target_locs:
            raise ValueError(f"Target field '{target}' was not found in dataset info.")

        validated_rules.append((source, target, scale, source_locs, target_locs))
        for column_name, _ in source_locs + target_locs:
            involved_columns.add(column_name)

    original_vectors: Dict[str, List[List[float] | None]] = {}
    transformed_vectors: Dict[str, List[List[float] | None]] = {}

    for column_name in involved_columns:
        column_vectors = table.column(column_name).combine_chunks().to_pylist()
        original_vectors[column_name] = [
            list(vector) if vector is not None else None for vector in column_vectors
        ]
        transformed_vectors[column_name] = [
            list(vector) if vector is not None else None for vector in column_vectors
        ]

    for source, target, scale, source_locs, target_locs in validated_rules:
        LOGGER.info("Applying joint rule: %s -> %s (scale=%s)", source, target, scale)
        for source_column, source_idx in source_locs:
            for target_column, target_idx in target_locs:
                source_rows = original_vectors[source_column]
                target_rows = transformed_vectors[target_column]

                for row_idx, source_vector in enumerate(source_rows):
                    if source_vector is None:
                        continue

                    source_value = source_vector[source_idx]
                    if source_value is None:
                        continue

                    target_vector = target_rows[row_idx]
                    if target_vector is None:
                        continue

                    target_vector[target_idx] = float(source_value) * scale

    output_arrays = []
    for column_name in table.column_names:
        if column_name in transformed_vectors:
            field = table.schema.field(column_name)
            output_arrays.append(pa.array(transformed_vectors[column_name], type=field.type))
        else:
            output_arrays.append(table.column(column_name).combine_chunks())

    return pa.Table.from_arrays(output_arrays, schema=table.schema)


def transform_parquet_dataset(
    input_root: Path,
    output_root: Path,
    info: dict,
    joint_rule_config: Path,
    rule_name: str | None,
) -> int:
    field_map = build_field_map(info)
    rules = load_joint_rules(joint_rule_config, rule_name)

    parquet_files = sorted((input_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files were found under: {input_root / 'data'}")

    processed = 0
    for parquet_path in parquet_files:
        table = read_parquet_table(parquet_path)
        mirrored_table = transform_table_with_rules(table, field_map, rules)
        output_path = output_root / parquet_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(mirrored_table, output_path)
        processed += 1

    return processed


def process_images(input_root: Path, output_root: Path, image_keys: Sequence[str]) -> int:
    paired_keys, independent_keys = build_image_plan(image_keys)
    total_images = 0

    for key_a, key_b in paired_keys:
        src_a = input_root / "images" / key_a
        src_b = input_root / "images" / key_b
        dst_a = output_root / "images" / key_a
        dst_b = output_root / "images" / key_b

        LOGGER.info("Mirroring and swapping image keys: %s <-> %s", key_a, key_b)
        total_images += mirror_tree(src_a, dst_b)
        total_images += mirror_tree(src_b, dst_a)

    for key in independent_keys:
        src_root = input_root / "images" / key
        dst_root = output_root / "images" / key
        LOGGER.info("Mirroring image key in place: %s", key)
        total_images += mirror_tree(src_root, dst_root)

    return total_images


def main() -> int:
    configure_logging()
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else build_default_output_root(input_root)
    )

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    info = load_dataset_info(input_root)
    image_keys = get_image_keys(info)

    ensure_output_root(output_root, args.overwrite)

    skip_data = args.mode == "full"
    copy_non_image_assets(input_root, output_root, skip_data=skip_data)

    LOGGER.info("Input dataset: %s", input_root)
    LOGGER.info("Output dataset: %s", output_root)
    LOGGER.info("Mode: %s", args.mode)
    LOGGER.info("Discovered image keys: %s", ", ".join(image_keys))

    processed_images = process_images(input_root, output_root, image_keys)
    LOGGER.info("Finished mirroring %d image files.", processed_images)

    if args.mode == "full":
        if not args.joint_rule_config:
            raise ValueError(
                "--joint-rule-config is required when --mode full is used."
            )

        joint_rule_config = Path(args.joint_rule_config).expanduser().resolve()
        if not joint_rule_config.exists():
            raise FileNotFoundError(
                f"Joint rule config does not exist: {joint_rule_config}"
            )

        processed_parquet = transform_parquet_dataset(
            input_root=input_root,
            output_root=output_root,
            info=info,
            joint_rule_config=joint_rule_config,
            rule_name=args.rule_name,
        )
        LOGGER.info("Finished transforming %d parquet files.", processed_parquet)

    LOGGER.info("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
