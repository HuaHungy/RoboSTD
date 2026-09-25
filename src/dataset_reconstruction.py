"""Dataset-level materialization for RoboSTD Stage 2."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

from src.robostd_stage2 import (
    CoordinationConstraints,
    build_planner,
    compose_stage_language,
    decompose_units,
    parse_constraints_dict,
    reconstruct_pseudo_bimanual,
)

logger = logging.getLogger(__name__)

STATE_COL = "observation.state"
ACTION_COL = "action"


def _read_vectors(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise KeyError(f"missing required column {column!r}")
    values = frame[column].values
    if len(values) == 0:
        raise ValueError("episode parquet is empty")
    try:
        array = np.vstack(values).astype(np.float64)
    except Exception as exc:
        raise ValueError(f"column {column!r} is not a fixed-length vector column") from exc
    if array.ndim != 2 or array.shape[1] % 2:
        raise ValueError(f"column {column!r} must have an even vector width")
    return array


def reconstruct_dataframe(
    original: pd.DataFrame,
    mirrored: pd.DataFrame,
    result,
    language: str,
    fps: Optional[float] = None,
    global_index_start: int = 0,
) -> pd.DataFrame:
    """Align scalar observations with the reconstructed state/action timeline."""
    rows = []
    for metadata in result.frame_metadata:
        source = original if metadata["observation_primary_source"] == "original" else mirrored
        frame_index = int(metadata["observation_frame"])
        if frame_index >= len(source):
            raise IndexError(f"observation frame {frame_index} exceeds source length {len(source)}")
        rows.append(source.iloc[frame_index].copy())
    output = pd.DataFrame(rows).reset_index(drop=True)
    output[STATE_COL] = [np.asarray(row, dtype=np.float32) for row in result.state]
    output[ACTION_COL] = [np.asarray(row, dtype=np.float32) for row in result.action]

    if "frame_index" in output:
        output["frame_index"] = np.arange(len(output), dtype=np.int64)
    if "index" in output:
        output["index"] = np.arange(
            global_index_start, global_index_start + len(output), dtype=np.int64
        )
    if "timestamp" in output:
        if fps and fps > 0:
            output["timestamp"] = np.arange(len(output), dtype=np.float32) / float(fps)
        elif len(output) > 1:
            values = np.asarray(output["timestamp"], dtype=np.float64)
            positive = np.diff(np.sort(np.unique(values)))
            step = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1.0
            output["timestamp"] = np.arange(len(output), dtype=np.float32) * step

    output["robostd.stage_id"] = [item["stage_id"] for item in result.frame_metadata]
    output["robostd.active_unit_left"] = [
        -1 if item["active_unit_left"] is None else item["active_unit_left"]
        for item in result.frame_metadata
    ]
    output["robostd.active_unit_right"] = [
        -1 if item["active_unit_right"] is None else item["active_unit_right"]
        for item in result.frame_metadata
    ]
    output["robostd.source_frame_left"] = [
        item["source_frame_left"] for item in result.frame_metadata
    ]
    output["robostd.source_frame_right"] = [
        item["source_frame_right"] for item in result.frame_metadata
    ]
    output["robostd.observation_source"] = [
        item["observation_source"] for item in result.frame_metadata
    ]
    output["robostd.stage_annotation"] = [
        item["stage_annotation"] for item in result.frame_metadata
    ]
    output["robostd.language"] = [
        compose_stage_language(language, item["stage_annotation"])
        for item in result.frame_metadata
    ]
    return output


def _read_all_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no readable frames: {path}")
    return frames, fps


def reconstruct_video(
    original_path: Path,
    mirrored_path: Path,
    output_path: Path,
    frame_metadata: Sequence[Mapping],
) -> None:
    """Materialize the primary aligned observation stream for one camera."""
    original_frames, original_fps = _read_all_frames(original_path)
    mirrored_frames, mirrored_fps = _read_all_frames(mirrored_path)
    height, width = original_frames[0].shape[:2]
    fps = original_fps or mirrored_fps or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video {output_path}")
    try:
        for item in frame_metadata:
            frames = (
                original_frames
                if item["observation_primary_source"] == "original"
                else mirrored_frames
            )
            index = int(item["observation_frame"])
            if index >= len(frames):
                raise IndexError(f"video frame {index} exceeds {len(frames)} in {output_path.name}")
            frame = frames[index]
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _episode_number(path: Path) -> int:
    match = re.search(r"episode_(\d+)$", path.stem)
    if not match:
        raise ValueError(f"cannot parse episode index from {path.name}")
    return int(match.group(1))


def _copy_dataset_shell(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.resolve() in {input_root.resolve(), input_root.parent.resolve()}:
        raise ValueError("output dataset must be separate from the input dataset")
    if output_root.exists():
        if not overwrite:
            if any(output_root.iterdir()):
                raise FileExistsError(f"output directory is not empty: {output_root}")
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for child in input_root.iterdir():
        if child.name in {"data", "videos", "images", "meta"}:
            continue
        destination = output_root / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)
    if (input_root / "meta").exists():
        shutil.copytree(input_root / "meta", output_root / "meta", dirs_exist_ok=True)


def reconstruct_lerobot_dataset(
    input_dir,
    mirror_dir,
    output_dir,
    *,
    n_units: int = 4,
    source_arm: str = "R",
    language: str = "",
    planner_name: str = "openai",
    model: str = "gpt-4.1",
    objects=None,
    workspace=None,
    action_mode: str = "position",
    allow_planner_fallback: bool = False,
    constraints_payload: Optional[Mapping] = None,
    unit_names: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> bool:
    """Build a complete pseudo-bimanual LeRobot dataset and aligned videos."""
    input_root = Path(input_dir)
    mirror_root = Path(mirror_dir)
    output_root = Path(output_dir)
    if not input_root.exists() or not mirror_root.exists():
        raise FileNotFoundError("both original and mirrored dataset roots must exist")
    _copy_dataset_shell(input_root, output_root, overwrite)

    info_path = input_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    fps = float(info.get("fps", 0) or 0)
    episode_records = {
        int(row["episode_index"]): row
        for row in _read_jsonl(input_root / "meta" / "episodes.jsonl")
    }
    task_by_episode = {
        episode_id: (record.get("tasks") or [""])[0]
        for episode_id, record in episode_records.items()
    }

    planner = build_planner(
        planner_name,
        source_arm=source_arm,
        model=model,
        allow_fallback=allow_planner_fallback,
    )
    task_indices: Dict[str, int] = {}
    output_episode_records: List[Dict] = []
    total_frames = 0
    total_videos = 0

    parquet_files = sorted((input_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise ValueError(f"no episode parquet files found under {input_root / 'data'}")

    for episode_path in parquet_files:
        relative = episode_path.relative_to(input_root)
        mirrored_path = mirror_root / relative
        if not mirrored_path.exists():
            raise FileNotFoundError(f"missing mirrored episode {mirrored_path}")
        original_df = pd.read_parquet(episode_path)
        mirrored_df = pd.read_parquet(mirrored_path)
        original_state = _read_vectors(original_df, STATE_COL)
        mirrored_state = _read_vectors(mirrored_df, STATE_COL)
        original_action = _read_vectors(original_df, ACTION_COL)
        mirrored_action = _read_vectors(mirrored_df, ACTION_COL)
        if len(original_df) != len(mirrored_df):
            raise ValueError(f"frame mismatch for {episode_path.name}")

        units = decompose_units(len(original_df), n_units=n_units, unit_names=unit_names)
        task_language = language or task_by_episode.get(_episode_number(episode_path), "")
        context = {"language": task_language, "objects": objects, "workspace": workspace}
        constraints: CoordinationConstraints
        if constraints_payload is not None:
            constraints = parse_constraints_dict(
                constraints_payload, [unit.id for unit in units]
            )
        else:
            constraints = planner.generate_constraints(units, context, source_arm)
        result = reconstruct_pseudo_bimanual(
            original_state,
            mirrored_state,
            original_action,
            mirrored_action,
            constraints,
            units,
            source_arm,
            action_mode,
        )
        output_df = reconstruct_dataframe(
            original_df,
            mirrored_df,
            result,
            task_language,
            fps=fps,
            global_index_start=total_frames,
        )
        for coordinated_language in output_df["robostd.language"].unique():
            if coordinated_language not in task_indices:
                task_indices[coordinated_language] = len(task_indices)
        if "task_index" in output_df:
            output_df["task_index"] = [
                task_indices[value] for value in output_df["robostd.language"]
            ]

        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_path, engine="pyarrow", index=False)

        episode_id = _episode_number(episode_path)
        plan_path = output_root / "meta" / "robostd_plans" / f"episode_{episode_id:06d}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(
                {
                    "episode_index": episode_id,
                    "source_arm": source_arm,
                    "action_mode": action_mode,
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

        for original_video in sorted(
            input_root.glob(f"videos/{episode_path.parent.name}/*/{episode_path.stem}.mp4")
        ):
            video_relative = original_video.relative_to(input_root)
            mirrored_video = mirror_root / video_relative
            if not mirrored_video.exists():
                raise FileNotFoundError(f"missing mirrored video {mirrored_video}")
            reconstruct_video(
                original_video,
                mirrored_video,
                output_root / video_relative,
                result.frame_metadata,
            )
            total_videos += 1

        coordinated_tasks = list(dict.fromkeys(output_df["robostd.language"].tolist()))
        record = dict(episode_records.get(episode_id, {"episode_index": episode_id}))
        record["length"] = len(output_df)
        record["tasks"] = coordinated_tasks
        output_episode_records.append(record)
        total_frames += len(output_df)
        logger.info(
            "Reconstructed %s: %d -> %d frames", episode_path.name, len(original_df), len(output_df)
        )

    meta_root = output_root / "meta"
    _write_jsonl(
        meta_root / "episodes.jsonl",
        sorted(output_episode_records, key=lambda row: row["episode_index"]),
    )
    _write_jsonl(
        meta_root / "tasks.jsonl",
        ({"task_index": index, "task": task} for task, index in task_indices.items()),
    )
    stale_stats = meta_root / "episodes_stats.jsonl"
    if stale_stats.exists():
        stale_stats.unlink()

    features = info.setdefault("features", {})
    for name in (
        "robostd.stage_id",
        "robostd.active_unit_left",
        "robostd.active_unit_right",
        "robostd.source_frame_left",
        "robostd.source_frame_right",
    ):
        features[name] = {"dtype": "int64", "shape": [1], "names": None}
    for name in (
        "robostd.observation_source",
        "robostd.stage_annotation",
        "robostd.language",
    ):
        features[name] = {"dtype": "string", "shape": [1], "names": None}
    info.update(
        {
            "total_episodes": len(output_episode_records),
            "total_frames": total_frames,
            "total_tasks": len(task_indices),
            "total_videos": total_videos,
            "robostd": {
                "stage": 2,
                "planner": planner_name,
                "model": model if planner_name == "openai" else None,
                "source_arm": source_arm,
                "action_mode": action_mode,
                "visual_alignment": (
                    "primary source stream; frames with simultaneous original and mirrored "
                    "skills are marked as mixed in robostd.observation_source"
                ),
            },
        }
    )
    (meta_root / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return True
