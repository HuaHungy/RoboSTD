import re
from pathlib import Path
from typing import List

import h5py
import numpy as np


def swap_left_right_name(name: str) -> str:
    s = re.sub(r"left", "__ROBOSTD_LR_TMP__", name, flags=re.IGNORECASE)
    s = re.sub(r"right", "left", s, flags=re.IGNORECASE)
    s = re.sub(r"__ROBOSTD_LR_TMP__", "right", s)
    return s


def _camera_swap_candidates(name: str) -> List[str]:
    if name == "left_wrist":
        return ["right_wrist", "right"]
    if name == "right_wrist":
        return ["left_wrist", "left"]
    if name == "left":
        return ["right", "right_wrist"]
    if name == "right":
        return ["left", "left_wrist"]

    swapped = swap_left_right_name(name)
    if swapped != name:
        return [swapped]
    return []


def _dataset_create_kwargs(template: h5py.Dataset) -> dict:
    kwargs = {}

    if template.chunks is not None:
        kwargs["chunks"] = template.chunks

    if template.compression is not None:
        kwargs["compression"] = template.compression
        if template.compression_opts is not None:
            kwargs["compression_opts"] = template.compression_opts

    if template.shuffle:
        kwargs["shuffle"] = True

    if template.fletcher32:
        kwargs["fletcher32"] = True

    if template.scaleoffset is not None:
        kwargs["scaleoffset"] = template.scaleoffset

    if template.fillvalue is not None:
        kwargs["fillvalue"] = template.fillvalue

    return kwargs


def mirror_aloha_14(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] != 14:
        raise ValueError(f"aloha14 expects last dim=14, got shape {x.shape}")

    within = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    right = x[..., :7].astype(np.float64, copy=False)
    left = x[..., 7:14].astype(np.float64, copy=False)

    out_first = left * within
    out_second = right * within
    out = np.concatenate([out_first, out_second], axis=-1)

    if np.issubdtype(x.dtype, np.floating) and x.dtype != np.float64:
        return out.astype(x.dtype)
    if np.issubdtype(x.dtype, np.integer):
        return np.rint(out).astype(x.dtype)
    return out


def mirror_act_episode_hdf5(
    input_h5: Path,
    output_h5: Path,
    joint_mode: str = "aloha14",
    swap_cameras: bool = True,
    frame_batch: int = 16,
) -> None:
    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as fin, h5py.File(output_h5, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        for k in fin.keys():
            if k in {"action", "observations", "is_pad"}:
                continue
            fin.copy(k, fout)

        if "is_pad" in fin:
            fin.copy("is_pad", fout)

        if "action" in fin:
            action = fin["action"][...]
            if joint_mode == "aloha14":
                action_out = mirror_aloha_14(action)
            else:
                raise ValueError(f"Unsupported joint_mode: {joint_mode}")
            tmpl = fin["action"]
            dset = fout.create_dataset(
                "action",
                data=action_out,
                dtype=action_out.dtype,
                **_dataset_create_kwargs(tmpl),
            )
            for ak, av in tmpl.attrs.items():
                dset.attrs[ak] = av

        if "observations" not in fin:
            raise KeyError("Missing group: observations")

        obs_in = fin["observations"]
        obs_out = fout.create_group("observations")
        for k, v in obs_in.attrs.items():
            obs_out.attrs[k] = v

        for k in obs_in.keys():
            if k in {"images", "qpos", "qvel"}:
                continue
            obs_in.copy(k, obs_out)

        for vec_name in ("qpos", "qvel"):
            if vec_name not in obs_in:
                continue
            x = obs_in[vec_name][...]
            if joint_mode == "aloha14":
                y = mirror_aloha_14(x)
            else:
                raise ValueError(f"Unsupported joint_mode: {joint_mode}")
            tmpl = obs_in[vec_name]
            dset = obs_out.create_dataset(
                vec_name,
                data=y,
                dtype=y.dtype,
                **_dataset_create_kwargs(tmpl),
            )
            for ak, av in tmpl.attrs.items():
                dset.attrs[ak] = av

        if "images" not in obs_in:
            raise KeyError("Missing group: observations/images")

        imgs_in = obs_in["images"]
        imgs_out = obs_out.create_group("images")
        for k, v in imgs_in.attrs.items():
            imgs_out.attrs[k] = v

        in_keys = list(imgs_in.keys())
        out_keys = in_keys

        out_to_in = {}
        for out_k in out_keys:
            if not swap_cameras:
                out_to_in[out_k] = out_k
                continue

            chosen = None
            for cand in _camera_swap_candidates(out_k):
                if cand in imgs_in:
                    chosen = cand
                    break
            out_to_in[out_k] = chosen if chosen is not None else out_k

        for out_k in sorted(out_to_in.keys()):
            in_k = out_to_in[out_k]
            d_in = imgs_in[in_k]
            if d_in.ndim != 4:
                raise ValueError(f"Expected images dataset ndim=4, got {in_k} ndim={d_in.ndim}")

            T, H, W, C = d_in.shape
            d_out = imgs_out.create_dataset(
                out_k,
                shape=d_in.shape,
                dtype=d_in.dtype,
                **_dataset_create_kwargs(d_in),
            )
            for ak, av in d_in.attrs.items():
                d_out.attrs[ak] = av

            if T == 0:
                continue

            batch = max(1, int(frame_batch))
            for t0 in range(0, T, batch):
                t1 = min(T, t0 + batch)
                frames = d_in[t0:t1]
                frames = frames[:, :, ::-1, :]
                d_out[t0:t1] = frames
