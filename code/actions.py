from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VehicleAction:
    throttle: float
    steer: float
    brake: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.throttle, self.steer, self.brake


GENTLE_STEERS = (-0.10, -0.03, -0.01, 0.0, 0.01, 0.03, 0.10)
ALL_STEERS = (-0.30, -0.10, -0.03, -0.01, 0.0, 0.01, 0.03, 0.10, 0.30)
SHARP_STEERS = (-0.30, 0.30)

# Longitudinal commands are mutually exclusive throttle-brake pairs.
NON_BRAKE_LONGITUDINAL = (
    (0.0, 0.0),
    (0.1, 0.0),
    (0.8, 0.0),
)
BRAKE_LONGITUDINAL = (
    (0.0, 0.04),
    (0.0, 0.20),
)


def _combine(longitudinal: Iterable[tuple[float, float]], steers: Iterable[float]) -> list[VehicleAction]:
    return [VehicleAction(throttle=t, steer=s, brake=b) for t, b in longitudinal for s in steers]


def build_actions() -> list[VehicleAction]:
    """Return a fixed 45-action head ordered so prefixes match curriculum stages.

    Indices 0:21  -> Stage I
    Indices 0:27  -> Stage II
    Indices 0:45  -> Stage III
    """
    stage1 = _combine(NON_BRAKE_LONGITUDINAL, GENTLE_STEERS)
    stage2_additions = _combine(NON_BRAKE_LONGITUDINAL, SHARP_STEERS)
    stage3_additions = _combine(BRAKE_LONGITUDINAL, ALL_STEERS)
    actions = stage1 + stage2_additions + stage3_additions
    if len(actions) != 45 or len(set(actions)) != 45:
        raise RuntimeError("The action construction must yield 45 unique actions.")
    return actions


ACTIONS = build_actions()
STAGE_ACTION_COUNTS = {1: 21, 2: 27, 3: 45}


def admissible_action_count(stage: int) -> int:
    try:
        return STAGE_ACTION_COUNTS[stage]
    except KeyError as exc:
        raise ValueError(f"Invalid curriculum stage: {stage}") from exc


def action_mask(stage: int) -> list[bool]:
    count = admissible_action_count(stage)
    return [i < count for i in range(len(ACTIONS))]
