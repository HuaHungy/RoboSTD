"""URDF-based sagittal reflection utilities for RoboSTD Stage 1.

The existing table-driven Stage 1 remains supported.  These helpers derive or
verify its joint signs from common-frame URDF axes as described by Eq. (7) of
the paper, avoiding unverified platform-specific sign tables.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def reflection_matrix(normal: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Return ``S = I - 2 n n^T`` for a plane through the frame origin."""
    vector = np.asarray(normal, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("reflection normal must contain three finite values")
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("reflection normal must be non-zero")
    vector /= norm
    return np.eye(3, dtype=np.float64) - 2.0 * np.outer(vector, vector)


def _vector(text: Optional[str], default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    values = [float(item) for item in text.split()]
    if len(values) != 3:
        raise ValueError(f"expected three values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_transform(origin: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    transform[:3, :3] = _rpy_matrix(_vector(origin.get("rpy"), (0, 0, 0)))
    transform[:3, 3] = _vector(origin.get("xyz"), (0, 0, 0))
    return transform


def joint_axes_in_common_frame(urdf_path: Path | str) -> Dict[str, np.ndarray]:
    """Resolve every non-fixed joint axis in the URDF root frame at q=0."""
    root = ET.parse(str(urdf_path)).getroot()
    joints = []
    child_links = set()
    all_links = {element.get("name") for element in root.findall("link")}
    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        child_links.add(child_link)
        joints.append(
            {
                "name": element.get("name"),
                "type": element.get("type", "fixed"),
                "parent": parent_link,
                "child": child_link,
                "origin": _origin_transform(element.find("origin")),
                "axis": _vector(
                    element.find("axis").get("xyz") if element.find("axis") is not None else None,
                    (1, 0, 0),
                ),
            }
        )

    roots = sorted(link for link in all_links if link and link not in child_links)
    if not roots:
        raise ValueError("URDF has no root link")
    link_transform = {link: np.eye(4, dtype=np.float64) for link in roots}
    unresolved = list(joints)
    axes: Dict[str, np.ndarray] = {}
    while unresolved:
        progress = False
        for joint in list(unresolved):
            if joint["parent"] not in link_transform:
                continue
            joint_transform = link_transform[joint["parent"]] @ joint["origin"]
            link_transform[joint["child"]] = joint_transform
            if joint["type"] not in {"fixed", "floating", "planar"}:
                axis = joint_transform[:3, :3] @ joint["axis"]
                norm = np.linalg.norm(axis)
                if norm <= 0:
                    raise ValueError(f"joint {joint['name']} has a zero axis")
                axes[str(joint["name"])] = axis / norm
            unresolved.remove(joint)
            progress = True
        if not progress:
            names = [joint["name"] for joint in unresolved]
            raise ValueError(f"could not resolve URDF joint tree for {names}")
    return axes


def derive_joint_signs_from_axes(
    source_axes: Sequence[Sequence[float]],
    target_axes: Sequence[Sequence[float]],
    normal: Sequence[float] = (0.0, 1.0, 0.0),
    tolerance: float = 1e-4,
) -> List[int]:
    """Derive Lambda using ``v_target = -S v_source`` or ``S v_source``.

    Returns ``+1`` for the first relation and ``-1`` for the second relation.
    """
    if len(source_axes) != len(target_axes):
        raise ValueError("source and target axis counts must match")
    reflection = reflection_matrix(normal)
    signs: List[int] = []
    for index, (source, target) in enumerate(zip(source_axes, target_axes)):
        source_axis = np.asarray(source, dtype=np.float64)
        target_axis = np.asarray(target, dtype=np.float64)
        source_axis /= np.linalg.norm(source_axis)
        target_axis /= np.linalg.norm(target_axis)
        reflected = reflection @ source_axis
        dot = float(np.dot(target_axis, reflected))
        if abs(abs(dot) - 1.0) > tolerance:
            raise ValueError(
                f"joint pair {index} is not mirror-compatible: axis dot product={dot:.6f}"
            )
        signs.append(-1 if dot > 0 else 1)
    return signs


def derive_joint_signs_from_urdf(
    urdf_path: Path | str,
    source_joints: Sequence[str],
    target_joints: Sequence[str],
    normal: Sequence[float] = (0.0, 1.0, 0.0),
    tolerance: float = 1e-4,
) -> List[int]:
    """Derive the joint-sign diagonal for named URDF joint pairs."""
    if len(source_joints) != len(target_joints):
        raise ValueError("source and target joint counts must match")
    axes = joint_axes_in_common_frame(urdf_path)
    missing = [name for name in [*source_joints, *target_joints] if name not in axes]
    if missing:
        raise KeyError(f"joints not found or fixed in URDF: {sorted(set(missing))}")
    return derive_joint_signs_from_axes(
        [axes[name] for name in source_joints],
        [axes[name] for name in target_joints],
        normal,
        tolerance,
    )


def build_bidirectional_rules(
    left_fields: Sequence[str],
    right_fields: Sequence[str],
    signs_left_to_right: Sequence[int],
) -> List[Dict[str, object]]:
    """Create Stage 1 YAML rules after geometry-based sign derivation."""
    if not (len(left_fields) == len(right_fields) == len(signs_left_to_right)):
        raise ValueError("left fields, right fields, and signs must have the same length")
    rules: List[Dict[str, object]] = []
    for left, right, sign in zip(left_fields, right_fields, signs_left_to_right):
        if int(sign) not in {-1, 1}:
            raise ValueError("joint signs must be +1 or -1")
        rules.append({"source": left, "target": right, "scale": float(sign)})
        rules.append({"source": right, "target": left, "scale": float(sign)})
    return rules


def verify_rule_signs(
    rules: Sequence[Mapping[str, object]],
    expected: Mapping[Tuple[str, str], int],
) -> None:
    """Raise when a table-driven Stage 1 rule disagrees with URDF geometry."""
    actual = {
        (str(rule["source"]), str(rule["target"])): int(float(rule.get("scale", 1)))
        for rule in rules
    }
    errors = []
    for pair, sign in expected.items():
        if pair not in actual:
            errors.append(f"missing rule {pair[0]} -> {pair[1]}")
        elif actual[pair] != int(sign):
            errors.append(
                f"rule {pair[0]} -> {pair[1]} uses {actual[pair]}, expected {int(sign)}"
            )
    if errors:
        raise ValueError("mirror-rule verification failed: " + "; ".join(errors))
