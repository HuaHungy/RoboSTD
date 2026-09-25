"""RoboSTD Stage 2: LLM-guided spatio-temporal reconstruction.

This module implements Section IV-B and Algorithm 1 of the RoboSTD paper.  It
keeps the original and sagittal-mirrored skills as immutable sources, validates
``G = (rho, E_pre, E_conf)``, schedules manipulation units on a common timeline,
holds inactive arms, and emits per-frame language/provenance annotations.

State and action rows use ``[left arm, right arm]`` ordering.  Any even feature
dimension is supported (for example ACT ALOHA-14 and the 16-dimensional sample
LeRobot data shipped with earlier revisions of this repository).
"""

from __future__ import annotations

import heapq
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ARMS: Tuple[str, str] = ("L", "R")
DEFAULT_N_UNITS = 4


@dataclass(frozen=True)
class ManipulationUnit:
    """A source-trajectory interval associated with one reusable skill."""

    id: int
    start: int
    end: int
    name: str = ""

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CoordinationConstraints:
    """Structured constraints ``G = (rho, E_pre, E_conf)``."""

    rho: Dict[int, str]
    e_pre: List[Tuple[int, int]]
    e_conf: List[Tuple[int, int]]


@dataclass(frozen=True)
class ScheduledUnit:
    """A manipulation unit placed on the reconstructed common timeline."""

    unit: ManipulationUnit
    arm: str
    start: int
    end: int
    skill_source: str


@dataclass
class ReconstructionResult:
    """Numerical supervision plus a reviewable schedule and frame provenance."""

    state: np.ndarray
    action: np.ndarray
    plan: List[Dict]
    frame_metadata: List[Dict]


COORDINATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rho": {
            "type": "object",
            "minProperties": 1,
            "propertyNames": {"pattern": "^[0-9]+$"},
            "additionalProperties": {"type": "string", "enum": ["L", "R"]},
        },
        "e_pre": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
        },
        "e_conf": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
        },
    },
    "required": ["rho", "e_pre", "e_conf"],
}

# OpenAI Structured Outputs requires closed object schemas.  A dynamic ``rho``
# mapping cannot be closed, so the wire representation uses assignment/edge
# records and is converted back to the paper notation by ``parse_constraints_dict``.
LLM_COORDINATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rho": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "unit_id": {"type": "integer", "minimum": 0},
                    "arm": {"type": "string", "enum": ["L", "R"]},
                },
                "required": ["unit_id", "arm"],
            },
        },
        "e_pre": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "before": {"type": "integer", "minimum": 0},
                    "after": {"type": "integer", "minimum": 0},
                },
                "required": ["before", "after"],
            },
        },
        "e_conf": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "a": {"type": "integer", "minimum": 0},
                    "b": {"type": "integer", "minimum": 0},
                },
                "required": ["a", "b"],
            },
        },
    },
    "required": ["rho", "e_pre", "e_conf"],
}


def _normalise_arm(arm: str) -> str:
    value = str(arm).upper()
    if value not in ARMS:
        raise ValueError(f"arm must be 'L' or 'R', got {arm!r}")
    return value


def decompose_units(
    n_frames: int,
    n_units: Optional[int] = None,
    unit_names: Optional[Sequence[str]] = None,
    boundaries: Optional[Sequence[int]] = None,
) -> List[ManipulationUnit]:
    """Create manipulation units from explicit boundaries or an equal split.

    Explicit boundaries are preferred because they preserve semantic skill
    boundaries.  Equal splitting remains as a backwards-compatible fallback.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")

    if boundaries is not None:
        bounds = [int(v) for v in boundaries]
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != n_frames:
            raise ValueError("boundaries must start at 0 and end at n_frames")
        if any(a >= b for a, b in zip(bounds, bounds[1:])):
            raise ValueError("boundaries must be strictly increasing")
        n = len(bounds) - 1
        if unit_names is not None and len(unit_names) != n:
            raise ValueError("unit_names count must match the boundary intervals")
    else:
        n = len(unit_names) if unit_names else int(n_units or DEFAULT_N_UNITS)
        if n <= 0 or n > n_frames:
            raise ValueError("n_units must be between 1 and n_frames")
        bounds = np.linspace(0, n_frames, n + 1, dtype=int).tolist()

    return [
        ManipulationUnit(
            id=i,
            start=int(bounds[i]),
            end=int(bounds[i + 1]),
            name=str(unit_names[i]) if unit_names else f"unit_{i}",
        )
        for i in range(n)
    ]


def parse_units_payload(payload: Sequence[Mapping], n_frames: int) -> List[ManipulationUnit]:
    """Parse semantic units supplied as JSON objects."""
    units = [
        ManipulationUnit(
            id=int(item["id"]),
            start=int(item["start"]),
            end=int(item["end"]),
            name=str(item.get("name", f"unit_{item['id']}")),
        )
        for item in payload
    ]
    validate_units(units, n_frames)
    return units


def validate_units(units: Sequence[ManipulationUnit], n_frames: int) -> None:
    if not units:
        raise ValueError("at least one manipulation unit is required")
    ids = [u.id for u in units]
    if len(ids) != len(set(ids)):
        raise ValueError("manipulation-unit ids must be unique")
    for unit in units:
        if unit.id < 0:
            raise ValueError(f"unit id must be non-negative: {unit.id}")
        if not (0 <= unit.start < unit.end <= n_frames):
            raise ValueError(
                f"unit {unit.id} has invalid interval [{unit.start}, {unit.end}) "
                f"for {n_frames} frames"
            )


def _parse_edges(value: object, field: str) -> List[Tuple[int, int]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    edges: List[Tuple[int, int]] = []
    seen = set()
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"each {field} entry must contain exactly two unit ids")
        if isinstance(item[0], bool) or isinstance(item[1], bool):
            raise ValueError(f"{field} ids must be integers")
        try:
            edge = (int(item[0]), int(item[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} ids must be integers") from exc
        if edge[0] == edge[1]:
            raise ValueError(f"{field} cannot contain self-edge {edge}")
        key = edge if field == "e_pre" else tuple(sorted(edge))
        if key in seen:
            raise ValueError(f"{field} contains duplicate edge {edge}")
        seen.add(key)
        edges.append(edge)
    return edges


def _find_cycle(unit_ids: Iterable[int], edges: Sequence[Tuple[int, int]]) -> bool:
    ids = set(unit_ids)
    indegree = {unit_id: 0 for unit_id in ids}
    successors = {unit_id: [] for unit_id in ids}
    for before, after in edges:
        successors[before].append(after)
        indegree[after] += 1
    ready = [unit_id for unit_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return visited != len(ids)


def validate_coordination(
    constraints: CoordinationConstraints,
    unit_ids: Optional[Iterable[int]] = None,
) -> List[str]:
    """Return all structural and semantic validation errors."""
    errors: List[str] = []
    rho_ids = set(constraints.rho)
    expected_ids = set(unit_ids) if unit_ids is not None else rho_ids

    if rho_ids != expected_ids:
        missing = sorted(expected_ids - rho_ids)
        extra = sorted(rho_ids - expected_ids)
        if missing:
            errors.append(f"rho is missing units {missing}")
        if extra:
            errors.append(f"rho references unknown units {extra}")
    for unit_id, arm in constraints.rho.items():
        if unit_id < 0:
            errors.append(f"rho contains negative unit id {unit_id}")
        if arm not in ARMS:
            errors.append(f"rho[{unit_id}]={arm!r} is not 'L' or 'R'")

    for field, edges in (("e_pre", constraints.e_pre), ("e_conf", constraints.e_conf)):
        for edge in edges:
            unknown = [unit_id for unit_id in edge if unit_id not in expected_ids]
            if unknown:
                errors.append(f"{field} edge {edge} references unknown units {unknown}")
            if edge[0] == edge[1]:
                errors.append(f"{field} contains self-edge {edge}")

    if not errors and _find_cycle(expected_ids, constraints.e_pre):
        errors.append("e_pre contains a directed cycle")
    return errors


def parse_constraints_dict(
    payload: Mapping,
    unit_ids: Optional[Iterable[int]] = None,
) -> CoordinationConstraints:
    """Strictly validate an LLM JSON object and return typed constraints."""
    if not isinstance(payload, Mapping):
        raise ValueError("coordination output must be a JSON object")
    expected_keys = set(COORDINATION_SCHEMA["required"])
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        raise ValueError(
            f"coordination output keys must be {sorted(expected_keys)}; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    raw_rho = payload["rho"]
    if isinstance(raw_rho, list):
        try:
            raw_rho = {str(item["unit_id"]): item["arm"] for item in raw_rho}
        except (KeyError, TypeError) as exc:
            raise ValueError("rho assignment records require unit_id and arm") from exc
        if len(raw_rho) != len(payload["rho"]):
            raise ValueError("rho contains duplicate unit assignments")
    if not isinstance(raw_rho, Mapping) or not raw_rho:
        raise ValueError("rho must be a non-empty object or assignment array")

    raw_pre = payload["e_pre"]
    if isinstance(raw_pre, list) and raw_pre and isinstance(raw_pre[0], Mapping):
        raw_pre = [[item["before"], item["after"]] for item in raw_pre]
    raw_conf = payload["e_conf"]
    if isinstance(raw_conf, list) and raw_conf and isinstance(raw_conf[0], Mapping):
        raw_conf = [[item["a"], item["b"]] for item in raw_conf]

    rho: Dict[int, str] = {}
    for raw_id, raw_arm in raw_rho.items():
        if isinstance(raw_id, bool) or not str(raw_id).isdigit():
            raise ValueError(f"rho key must be a non-negative integer string: {raw_id!r}")
        unit_id = int(raw_id)
        if unit_id in rho:
            raise ValueError(f"rho contains duplicate unit id after conversion: {unit_id}")
        rho[unit_id] = _normalise_arm(str(raw_arm))

    constraints = CoordinationConstraints(
        rho=rho,
        e_pre=_parse_edges(raw_pre, "e_pre"),
        e_conf=_parse_edges(raw_conf, "e_conf"),
    )
    errors = validate_coordination(constraints, unit_ids)
    if errors:
        raise ValueError("invalid coordination constraints: " + "; ".join(errors))
    return constraints


class BasePlanner:
    def generate_constraints(
        self,
        units: Sequence[ManipulationUnit],
        task_context: Optional[Dict] = None,
        source_arm: str = "R",
    ) -> CoordinationConstraints:
        raise NotImplementedError


class DefaultPlanner(BasePlanner):
    """Deterministic paper ablation: source arm and original unit order."""

    def __init__(self, source_arm: str = "R"):
        self.source_arm = _normalise_arm(source_arm)

    def generate_constraints(self, units, task_context=None, source_arm=None):
        arm = _normalise_arm(source_arm or self.source_arm)
        ordered = sorted(units, key=lambda unit: (unit.start, unit.id))
        return CoordinationConstraints(
            rho={unit.id: arm for unit in units},
            e_pre=[(a.id, b.id) for a, b in zip(ordered, ordered[1:])],
            e_conf=[],
        )


class OpenAIPlanner(BasePlanner):
    """GPT-4.1 planner with strict structured output validation."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        api_key: Optional[str] = None,
        allow_fallback: bool = False,
        default_source_arm: str = "R",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.allow_fallback = bool(allow_fallback)
        self.fallback = DefaultPlanner(default_source_arm)

    def _fallback_or_raise(self, message: str, units, task_context, source_arm, exc=None):
        if self.allow_fallback:
            logger.warning("%s; using deterministic w/o-LLM planner", message)
            return self.fallback.generate_constraints(units, task_context, source_arm)
        raise RuntimeError(message) from exc

    def _messages(self, units: Sequence[ManipulationUnit], task_context: Optional[Dict]):
        context = task_context or {}
        descriptions = "\n".join(
            f"- {unit.id}: {unit.name}; source frames [{unit.start}, {unit.end})"
            for unit in units
        )
        return [
            {
                "role": "system",
                "content": (
                    "Plan bimanual manipulation for arms L and R. Assign every supplied unit "
                    "exactly once in rho. Add directed precedence edges only when required. "
                    "Add conflict pairs when two units must not overlap because of workspace, "
                    "reachability, object-state, or safety constraints. The precedence graph "
                    "must be acyclic. Return only the requested structured JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction: {context.get('language') or '(none)'}\n"
                    f"Objects and locations: {context.get('objects') or '(none)'}\n"
                    f"Workspace, reachability, and safety: {context.get('workspace') or '(none)'}\n"
                    f"Available manipulation units:\n{descriptions}"
                ),
            },
        ]

    def generate_constraints(self, units, task_context=None, source_arm="R"):
        if not self.api_key:
            return self._fallback_or_raise(
                "OPENAI_API_KEY is required for the paper planner",
                units,
                task_context,
                source_arm,
            )
        try:
            from openai import OpenAI
        except Exception as exc:
            return self._fallback_or_raise(
                "the openai package is required for the paper planner",
                units,
                task_context,
                source_arm,
                exc,
            )

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=self._messages(units, task_context),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "robostd_coordination",
                        "strict": True,
                        "schema": LLM_COORDINATION_SCHEMA,
                    },
                },
                temperature=0.0,
            )
            payload = json.loads(response.choices[0].message.content)
            return parse_constraints_dict(payload, [unit.id for unit in units])
        except Exception as exc:
            return self._fallback_or_raise(
                f"LLM planning or schema validation failed: {exc}",
                units,
                task_context,
                source_arm,
                exc,
            )


def build_planner(
    name: str = "default",
    source_arm: str = "R",
    custom_planner: Optional[Callable] = None,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    allow_fallback: bool = False,
) -> BasePlanner:
    if custom_planner is not None:
        class _CustomPlanner(BasePlanner):
            def generate_constraints(self, units, task_context=None, source_arm="R"):
                result = custom_planner(units, task_context, source_arm)
                errors = validate_coordination(result, [unit.id for unit in units])
                if errors:
                    raise ValueError("invalid custom planner result: " + "; ".join(errors))
                return result

        return _CustomPlanner()
    if name == "openai":
        return OpenAIPlanner(
            model=model,
            api_key=api_key,
            allow_fallback=allow_fallback,
            default_source_arm=source_arm,
        )
    if name != "default":
        raise ValueError(f"unknown planner {name!r}")
    return DefaultPlanner(source_arm)


def schedule_units(
    units: Sequence[ManipulationUnit],
    constraints: CoordinationConstraints,
    source_arm: str = "R",
) -> List[ScheduledUnit]:
    """Apply precedence, arm-resource, and conflict constraints."""
    source_arm = _normalise_arm(source_arm)
    by_id = {unit.id: unit for unit in units}
    validate_units(units, max(unit.end for unit in units))
    errors = validate_coordination(constraints, by_id)
    if errors:
        raise ValueError("invalid coordination constraints: " + "; ".join(errors))

    predecessors = {unit_id: [] for unit_id in by_id}
    successors = {unit_id: [] for unit_id in by_id}
    indegree = {unit_id: 0 for unit_id in by_id}
    for before, after in constraints.e_pre:
        predecessors[after].append(before)
        successors[before].append(after)
        indegree[after] += 1

    ready: List[Tuple[int, int]] = []
    for unit_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (by_id[unit_id].start, unit_id))
    order: List[int] = []
    while ready:
        _, unit_id = heapq.heappop(ready)
        order.append(unit_id)
        for nxt in successors[unit_id]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, (by_id[nxt].start, nxt))
    if len(order) != len(units):
        raise ValueError("e_pre contains a directed cycle")

    conflicts = {unit_id: set() for unit_id in by_id}
    for left, right in constraints.e_conf:
        conflicts[left].add(right)
        conflicts[right].add(left)

    scheduled: Dict[int, ScheduledUnit] = {}
    arm_available = {arm: 0 for arm in ARMS}
    for unit_id in order:
        unit = by_id[unit_id]
        arm = constraints.rho[unit_id]
        start = arm_available[arm]
        if predecessors[unit_id]:
            start = max(start, max(scheduled[pred].end for pred in predecessors[unit_id]))

        # A later-scheduled conflicting unit is shifted after every conflicting
        # interval already placed on the common timeline.
        while True:
            overlapping = [
                scheduled[other]
                for other in conflicts[unit_id]
                if other in scheduled
                and start < scheduled[other].end
                and start + unit.duration > scheduled[other].start
            ]
            if not overlapping:
                break
            start = max(item.end for item in overlapping)

        end = start + unit.duration
        skill_source = "original" if arm == source_arm else "mirror"
        scheduled[unit_id] = ScheduledUnit(unit, arm, start, end, skill_source)
        arm_available[arm] = end

    return sorted(scheduled.values(), key=lambda item: (item.start, item.arm, item.unit.id))


def _validate_arrays(*arrays: np.ndarray) -> Tuple[List[np.ndarray], int, int]:
    converted = [np.asarray(array, dtype=np.float64) for array in arrays]
    if converted[0].ndim != 2:
        raise ValueError("state_orig must be a 2-D array")
    shape = converted[0].shape
    if shape[1] < 2 or shape[1] % 2:
        raise ValueError("state/action feature dimension must be even and at least 2")
    for array in converted[1:]:
        if array.ndim != 2 or array.shape != shape:
            raise ValueError(f"all state/action arrays must have shape {shape}")
    return converted, shape[0], shape[1] // 2


def _stage_annotation(stage_id: int, active: Dict[str, Optional[ScheduledUnit]]) -> str:
    roles = []
    for arm in ARMS:
        item = active[arm]
        role = "hold" if item is None else f"execute {item.unit.name}"
        roles.append(f"{arm} arm: {role}")
    return f"stage {stage_id}; " + "; ".join(roles)


def reconstruct_pseudo_bimanual(
    state_orig: np.ndarray,
    state_mir: np.ndarray,
    action_orig: np.ndarray,
    action_mir: np.ndarray,
    constraints: CoordinationConstraints,
    units: Optional[Sequence[ManipulationUnit]] = None,
    source_arm: str = "R",
    action_mode: str = "position",
) -> ReconstructionResult:
    """Compose original and mirrored skills on a constraint-aware timeline.

    ``action_mode='position'`` holds an inactive arm by repeating its current
    joint state.  ``action_mode='delta'`` uses zero, which is appropriate for
    velocity/delta commands.
    """
    (state_orig, state_mir, action_orig, action_mir), n_frames, arm_dim = _validate_arrays(
        state_orig, state_mir, action_orig, action_mir
    )
    source_arm = _normalise_arm(source_arm)
    if action_mode not in {"position", "delta"}:
        raise ValueError("action_mode must be 'position' or 'delta'")
    if units is None:
        units = decompose_units(n_frames, n_units=len(constraints.rho))
    units = list(units)
    validate_units(units, n_frames)
    schedule = schedule_units(units, constraints, source_arm)
    timeline_length = max(item.end for item in schedule)

    slices = {"L": slice(0, arm_dim), "R": slice(arm_dim, arm_dim * 2)}
    state_sources = {
        source_arm: state_orig,
        "L" if source_arm == "R" else "R": state_mir,
    }
    action_sources = {
        source_arm: action_orig,
        "L" if source_arm == "R" else "R": action_mir,
    }

    active_by_arm: Dict[str, List[Optional[ScheduledUnit]]] = {
        arm: [None] * timeline_length for arm in ARMS
    }
    for item in schedule:
        for timestep in range(item.start, item.end):
            if active_by_arm[item.arm][timestep] is not None:
                raise RuntimeError(f"arm {item.arm} received overlapping units")
            active_by_arm[item.arm][timestep] = item

    output_state = np.zeros((timeline_length, arm_dim * 2), dtype=np.float64)
    output_action = np.zeros_like(output_state)
    current_state = {
        arm: state_sources[arm][0, slices[arm]].copy() for arm in ARMS
    }
    last_source_frame = {arm: 0 for arm in ARMS}
    metadata: List[Dict] = []
    last_signature = None
    stage_id = -1
    primary_source = "original"
    primary_frame = 0

    for timestep in range(timeline_length):
        active = {arm: active_by_arm[arm][timestep] for arm in ARMS}
        signature = tuple(active[arm].unit.id if active[arm] else None for arm in ARMS)
        if signature != last_signature:
            stage_id += 1
            last_signature = signature

        for arm in ARMS:
            arm_slice = slices[arm]
            item = active[arm]
            if item is not None:
                source_frame = item.unit.start + (timestep - item.start)
                last_source_frame[arm] = source_frame
                current_state[arm] = state_sources[arm][source_frame, arm_slice].copy()
                output_action[timestep, arm_slice] = action_sources[arm][source_frame, arm_slice]
            elif action_mode == "position":
                output_action[timestep, arm_slice] = current_state[arm]
            output_state[timestep, arm_slice] = current_state[arm]

        active_items = [item for item in active.values() if item is not None]
        active_sources = {item.skill_source for item in active_items}
        if active_items:
            preferred = active[source_arm] or active_items[0]
            primary_source = preferred.skill_source
            primary_frame = preferred.unit.start + (timestep - preferred.start)
        observation_source = (
            "mixed" if len(active_sources) > 1 else next(iter(active_sources), primary_source)
        )
        annotation = _stage_annotation(stage_id, active)
        metadata.append(
            {
                "timeline_frame": timestep,
                "stage_id": stage_id,
                "active_unit_left": active["L"].unit.id if active["L"] else None,
                "active_unit_right": active["R"].unit.id if active["R"] else None,
                "source_frame_left": last_source_frame["L"],
                "source_frame_right": last_source_frame["R"],
                "observation_source": observation_source,
                "observation_primary_source": primary_source,
                "observation_frame": primary_frame,
                "stage_annotation": annotation,
            }
        )

    plan = [
        {
            "unit": item.unit.id,
            "name": item.unit.name,
            "source_frames": [item.unit.start, item.unit.end],
            "scheduled_frames": [item.start, item.end],
            "arm": item.arm,
            "skill_source": item.skill_source,
        }
        for item in schedule
    ]
    return ReconstructionResult(output_state, output_action, plan, metadata)


def rearrange_pseudo_bimanual(
    state_orig: np.ndarray,
    state_mir: np.ndarray,
    action_orig: np.ndarray,
    action_mir: np.ndarray,
    constraints: CoordinationConstraints,
    units: Optional[Sequence[ManipulationUnit]] = None,
    source_arm: str = "R",
    action_mode: str = "position",
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Backwards-compatible wrapper returning ``(state, action, plan)``."""
    result = reconstruct_pseudo_bimanual(
        state_orig,
        state_mir,
        action_orig,
        action_mir,
        constraints,
        units,
        source_arm,
        action_mode,
    )
    return result.state, result.action, result.plan


def compose_stage_language(base_instruction: str, annotation: str) -> str:
    """Implement ``ell_coord_t = ell + c_t`` with stable punctuation."""
    base = (base_instruction or "").strip()
    return f"{base}. {annotation}" if base else annotation
