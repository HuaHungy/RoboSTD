"""RoboSTD Stage 2: LLM-Guided Spatio-Temporal Reconstruction.

Implements the second stage of the RoboSTD method (Section IV-B of the paper).
Starting from a single-arm demonstration ``D_single`` and its sagittal-mirrored
contralateral version ``D~_single`` (produced by Stage 1), it composes the two
into pseudo-bimanual supervision using structured spatio-temporal constraints

    G = (rho, E_pre, E_conf) = LLM(V, ell, O, W) ,

where ``rho`` assigns each manipulation unit to an arm, ``E_pre`` encodes
precedence between units, and ``E_conf`` marks units that must not execute
simultaneously.  The rearrangement operator ``R`` then aligns both arm
sequences on a common (frame-preserving) timeline:

    D_bi^pseudo = R(D_single, D~_single ; G) .

Formats
-------
The CoRobot / ALOHA 14-DoF layout is assumed: ``observation.state`` and
``action`` are 14-dim rows ``[left(7), right(7)]``.  Stage 1 already mirrors
these together with camera observations (see ``act_hdf5_mirror.mirror_aloha_14``
and ``joint_processor``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# ALOHA 14-DoF joint layout
# --------------------------------------------------------------------------- #
LEFT_SLICE = slice(0, 7)
RIGHT_SLICE = slice(7, 14)
ARM_SLICE: Dict[str, slice] = {"L": LEFT_SLICE, "R": RIGHT_SLICE}
ARMS: Tuple[str, str] = ("L", "R")

DEFAULT_N_UNITS = 4  # matches the 4-stage sandwich-making example in the paper


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class ManipulationUnit:
    """A time segment of the (single-arm) trajectory associated with a skill."""

    id: int
    start: int  # inclusive frame index
    end: int  # exclusive frame index
    name: str = ""


@dataclass
class CoordinationConstraints:
    """Structured spatio-temporal coordination constraints G = (rho, E_pre, E_conf)."""

    rho: Dict[int, str]  # unit id -> 'L' / 'R'
    e_pre: List[Tuple[int, int]]  # (before, after): before must precede after
    e_conf: List[Tuple[int, int]]  # two units that must not execute simultaneously


# --------------------------------------------------------------------------- #
# JSON schema (used to auto-validate the LLM output before reconstruction)
# --------------------------------------------------------------------------- #
COORDINATION_SCHEMA = {
    "type": "object",
    "properties": {
        "rho": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": ["L", "R"]},
        },
        "e_pre": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [{"type": "integer"}, {"type": "integer"}],
            },
        },
        "e_conf": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [{"type": "integer"}, {"type": "integer"}],
            },
        },
    },
    "required": ["rho", "e_pre", "e_conf"],
}


def decompose_units(
    n_frames: int,
    n_units: Optional[int] = None,
    unit_names: Optional[Sequence[str]] = None,
) -> List[ManipulationUnit]:
    """Split a trajectory into ``n_units`` manipulation units (equal-length).

    If explicit ``unit_names`` are given and their count differs from
    ``n_units``, the number of units is taken from ``unit_names``.
    """
    if unit_names:
        n = len(unit_names)
    else:
        n = int(n_units) if n_units else DEFAULT_N_UNITS
    n = max(1, n)

    bounds = np.linspace(0, n_frames, n + 1, dtype=int)
    units: List[ManipulationUnit] = []
    for i in range(n):
        name = unit_names[i] if unit_names and i < len(unit_names) else ""
        units.append(ManipulationUnit(id=i, start=int(bounds[i]), end=int(bounds[i + 1]), name=name))
    return units


def validate_coordination(constraints: CoordinationConstraints) -> List[str]:
    """Structural sanity checks on G. Returns a list of warning strings."""
    warnings: List[str] = []
    for i, arm in constraints.rho.items():
        if arm not in ARMS:
            warnings.append(f"rho[{i}] = {arm!r} is not a valid arm ('L'/'R')")
    for a, b in constraints.e_pre:
        if a not in constraints.rho or b not in constraints.rho:
            warnings.append(f"e_pre references unknown unit: ({a}, {b})")
    for a, b in constraints.e_conf:
        if a not in constraints.rho or b not in constraints.rho:
            warnings.append(f"e_conf references unknown unit: ({a}, {b})")
    return warnings


# --------------------------------------------------------------------------- #
# Planners
# --------------------------------------------------------------------------- #
class BasePlanner:
    """Interface for producing coordination constraints from a task context."""

    def generate_constraints(
        self,
        units: Sequence[ManipulationUnit],
        task_context: Optional[Dict] = None,
        source_arm: str = "R",
    ) -> CoordinationConstraints:
        raise NotImplementedError


class DefaultPlanner(BasePlanner):
    """Deterministic fallback (no API required).

    Assigns every unit to ``source_arm`` with no explicit precedence or
    conflict constraints.  This matches the "RoboSTD w/o LLM" baseline of the
    paper: geometric mirroring + in-place rearrangement, but without the
    LLM-inferred arm assignment, precedence, or conflict constraints.
    """

    def __init__(self, source_arm: str = "R"):
        self.source_arm = source_arm

    def generate_constraints(
        self,
        units: Sequence[ManipulationUnit],
        task_context: Optional[Dict] = None,
        source_arm: Optional[str] = None,
    ) -> CoordinationConstraints:
        arm = source_arm or self.source_arm
        rho = {u.id: arm for u in units}
        return CoordinationConstraints(rho=rho, e_pre=[], e_conf=[])


class OpenAIPlanner(BasePlanner):
    """LLM planner using OpenAI GPT-4.1 (as in the paper).

    The ``openai`` package is imported lazily so the rest of RoboSTD remains
    usable without it.  The model returns a JSON object (validated against
    ``COORDINATION_SCHEMA``).  On any error the planner falls back to
    ``DefaultPlanner``, guaranteeing a deterministic result.
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        api_key: Optional[str] = None,
        fallback: Optional[BasePlanner] = None,
        default_source_arm: str = "R",
    ):
        self.model = model
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key
        self.fallback = fallback or DefaultPlanner(default_source_arm)
        self.default_source_arm = default_source_arm

    def _build_messages(self, units: Sequence[ManipulationUnit], task_context: Optional[Dict]) -> List[Dict]:
        tc = task_context or {}
        unit_lines = "\n".join(
            f"- unit {u.id}: frames [{u.start}, {u.end}){(' : ' + u.name) if u.name else ''}"
            for u in units
        )
        system = (
            "You are a bimanual manipulation planner for a two-arm ALOHA robot "
            "with arms 'L' and 'R'.  Given a single-arm manipulation task, decompose "
            "the manipulation units into spatio-temporal coordination constraints. "
            "Return ONLY valid JSON matching this schema:\n"
            f"{COORDINATION_SCHEMA}\n"
            "rho maps each unit id to the arm that should execute it ('L' or 'R'). "
            "e_pre lists directed (before, after) unit-id pairs. e_conf lists pairs of "
            "units that must never execute simultaneously (shared workspace or conflicts)."
        )
        user = (
            f"Task language instruction: {tc.get('language', '(none)')}\n"
            f"Objects/identities: {tc.get('objects', '(none)')}\n"
            f"Workspace/reachability/safety: {tc.get('workspace', '(none)')}\n"
            f"Manipulation units:\n{unit_lines}\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate_constraints(
        self,
        units: Sequence[ManipulationUnit],
        task_context: Optional[Dict] = None,
        source_arm: Optional[str] = None,
    ) -> CoordinationConstraints:
        if not self.api_key:
            logger.warning("No OPENAI_API_KEY set; using deterministic fallback planner.")
            return self.fallback.generate_constraints(units, task_context, source_arm)

        try:
            import json

            from openai import OpenAI
        except Exception as exc:  # missing dependency
            logger.warning(f"openai not available ({exc}); using deterministic fallback planner.")
            return self.fallback.generate_constraints(units, task_context, source_arm)

        try:
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(units, task_context),
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            payload = json.loads(resp.choices[0].message.content)
            constraints = parse_constraints_dict(payload)
            logger.info("LLM produced coordination constraints: %s", constraints)
            return constraints
        except Exception as exc:
            logger.warning(f"LLM planning failed ({exc}); using deterministic fallback planner.")
            return self.fallback.generate_constraints(units, task_context, source_arm)


def parse_constraints_dict(payload: Dict) -> CoordinationConstraints:
    """Convert an LLM JSON payload into validated ``CoordinationConstraints``."""
    missing = set(COORDINATION_SCHEMA["required"]) - set(payload.keys())
    if missing:
        raise ValueError(f"LLM output missing required keys: {sorted(missing)}")

    rho = {int(k): str(v) for k, v in payload["rho"].items()}
    e_pre = [(int(a), int(b)) for a, b in payload["e_pre"]]
    e_conf = [(int(a), int(b)) for a, b in payload["e_conf"]]
    constraints = CoordinationConstraints(rho=rho, e_pre=e_pre, e_conf=e_conf)

    warn = validate_coordination(constraints)
    if warn:
        logger.warning("Coordination-constraint warnings: %s", warn)
    for i, arm in rho.items():
        if arm not in ARMS:
            raise ValueError(f"rho[{i}] = {arm!r} invalid")
    return constraints


def build_planner(
    name: str = "default",
    source_arm: str = "R",
    custom_planner: Optional[Callable] = None,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
) -> BasePlanner:
    """Factory for a pluggable planner.

    ``custom_planner`` may be any callable with signature
    ``(units, task_context, source_arm) -> CoordinationConstraints``; it takes
    precedence.  Otherwise ``name`` selects 'default' (deterministic) or 'openai'.
    """

    if custom_planner is not None:
        class _CustomPlanner(BasePlanner):
            def generate_constraints(self, units, task_context=None, source_arm=None):
                return custom_planner(units, task_context, source_arm or "R")

        return _CustomPlanner()
    if name == "openai":
        return OpenAIPlanner(model=model, api_key=api_key, default_source_arm=source_arm)
    return DefaultPlanner(source_arm)


# --------------------------------------------------------------------------- #
# Rearrangement operator R
# --------------------------------------------------------------------------- #
def rearrange_pseudo_bimanual(
    state_orig: np.ndarray,
    state_mir: np.ndarray,
    action_orig: np.ndarray,
    action_mir: np.ndarray,
    constraints: CoordinationConstraints,
    units: Optional[Sequence[ManipulationUnit]] = None,
    source_arm: str = "R",
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Compose original + mirrored single-arm sequences into pseudo-bimanual supervision.

    Implements ``D_bi^pseudo = R(D_single, D~_single; G)``.

    Parameters
    ----------
    state_orig, state_mir : (T, 14) float arrays
        The original single-arm demonstration and its sagittal-mirrored version.
        ``source_arm`` is the arm that performs the skill in the original demo.
    action_orig, action_mir : (T, 14) float arrays
        Corresponding action arrays (same 14-DoF layout).
    constraints : CoordinationConstraints
        LLM / default coordination constraints G.
    units : optional
        Manipulation-unit segmentation.  If omitted, it is derived by an
        equal-length split consistent with ``constraints.rho``.
    source_arm : 'L' or 'R'
        Arm that is active in the original single-arm demonstration.

    Returns
    -------
    (bi_state, bi_action, plan)
        Pseudo-bimanual state/action arrays with the same length ``T`` and a
        per-unit plan describing arm assignment.  During a unit assigned to one
        arm, the other arm holds its current configuration (zero action).
    """
    state_orig = np.asarray(state_orig, dtype=np.float64)
    state_mir = np.asarray(state_mir, dtype=np.float64)
    action_orig = np.asarray(action_orig, dtype=np.float64)
    action_mir = np.asarray(action_mir, dtype=np.float64)

    T = state_orig.shape[0]
    for arr, name in (
        (state_mir, "state_mir"),
        (action_orig, "action_orig"),
        (action_mir, "action_mir"),
    ):
        if arr.ndim != 2 or arr.shape != (T, 14):
            raise ValueError(f"{name} must have shape (T={T}, 14)")

    source_arm = source_arm.upper()
    if source_arm not in ARMS:
        raise ValueError(f"source_arm must be 'L' or 'R', got {source_arm!r}")

    # Segmentation: match the unit ids referenced by rho.
    ids = set(constraints.rho.keys())
    if units is not None:
        units = [u for u in units if u.id in ids]
        units.sort(key=lambda u: u.id)
    else:
        n = max(ids, default=DEFAULT_N_UNITS - 1) + 1
        units = decompose_units(T, n_units=n)
    units_by_id = {u.id: u for u in units}

    # Skill source for each arm.
    #  - The source arm reads its (original) own slices from the original demo.
    #  - The contralateral arm reads its mirrored slices from the mirrored demo.
    skill_state: Dict[str, np.ndarray] = {
        "L": state_mir[:, LEFT_SLICE],
        "R": state_orig[:, RIGHT_SLICE],
    }
    skill_action: Dict[str, np.ndarray] = {
        "L": action_mir[:, LEFT_SLICE],
        "R": action_orig[:, RIGHT_SLICE],
    }

    # Map each frame to its unit, then to the assigned arm.
    unit_of: np.ndarray = np.empty(T, dtype=int)
    for u in units:
        unit_of[u.start : u.end] = u.id
    arm_of: Dict[int, str] = constraints.rho

    bi_state = np.zeros_like(state_orig, dtype=np.float64)
    bi_action = np.zeros_like(action_orig, dtype=np.float64)
    plan: List[Dict] = []

    for arm in ARMS:
        sl = ARM_SLICE[arm]
        # Start with the arm's configuration from its own skill trajectory.
        current_state = skill_state[arm][0].copy()
        for t in range(T):
            u = int(unit_of[t])
            if arm_of.get(u) == arm:
                current_state = skill_state[arm][t].copy()
                bi_state[t, sl] = current_state
                bi_action[t, sl] = skill_action[arm][t]
            else:
                # Inactive arm maintains its current configuration.
                bi_state[t, sl] = current_state
                bi_action[t, sl] = 0.0

    last_planned_start = -1
    for u in units:
        arm = arm_of.get(u.id, source_arm)
        plan.append(
            {
                "unit": u.id,
                "name": u.name,
                "frames": [int(u.start), int(u.end)],
                "arm": arm,
                "skill_source": "mirror" if arm != source_arm else "original",
            }
        )
        if u.start < last_planned_start:
            logger.warning("Unit %d out of temporal order - E_pre may be violated.", u.id)
        last_planned_start = u.start

    return bi_state, bi_action, plan