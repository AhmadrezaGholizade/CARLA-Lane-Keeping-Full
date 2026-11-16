from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import TrainingConfig


def _flag(info: dict, *keys: str) -> bool:
    return any(bool(info.get(key, False)) for key in keys)


def collision_from_info(info: dict) -> bool:
    return _flag(info, "collision", "collided", "has_collision")


def departure_from_info(info: dict) -> bool:
    return _flag(info, "lane_departure", "off_road", "road_departure")


def lateral_error_from_info(info: dict) -> float:
    for key in ("lateral_dist", "lateral_error", "distance_from_center"):
        if key in info:
            return float(info[key])
    raise KeyError("Info must contain lateral_dist/lateral_error/distance_from_center.")


def heading_error_from_info(info: dict) -> float:
    for key in ("angle", "heading_error", "heading_error_deg"):
        if key in info:
            return float(info[key])
    raise KeyError("Info must contain angle/heading_error/heading_error_deg.")


def route_completion_from_info(info: dict) -> float:
    for key in ("route_completion", "completion", "route_completion_ratio"):
        if key in info:
            return float(np.clip(info[key], 0.0, 1.0))
    return 0.0


@dataclass
class RewardOutput:
    total: float
    base: float
    heading_term: float
    lateral_term: float
    terminal_reason: str | None = None


class LaneKeepingReward:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg

    def compute(
        self,
        current_info: dict,
        previous_info: dict,
        speed_kmh: float,
        stalled: bool = False,
    ) -> RewardOutput:
        if collision_from_info(current_info) or departure_from_info(current_info):
            reason = "collision" if collision_from_info(current_info) else "lane_departure"
            return RewardOutput(
                total=self.cfg.collision_or_departure_penalty,
                base=0.0,
                heading_term=0.0,
                lateral_term=0.0,
                terminal_reason=reason,
            )
        if stalled:
            return RewardOutput(
                total=self.cfg.stall_penalty,
                base=0.0,
                heading_term=0.0,
                lateral_term=0.0,
                terminal_reason="stall",
            )

        lateral = abs(lateral_error_from_info(current_info))
        heading = abs(heading_error_from_info(current_info))
        previous_lateral = abs(lateral_error_from_info(previous_info))
        previous_heading = abs(heading_error_from_info(previous_info))

        # The target speed is fixed at 50 km/h and is not a network input.
        base = 25.0 - abs(float(speed_kmh) - self.cfg.target_speed_kmh) / 2.0

        if lateral <= self.cfg.well_aligned_lateral_m and heading <= self.cfg.well_aligned_heading_deg:
            heading_term = 0.0
            lateral_term = 0.0
        else:
            # Manuscript Eqs. (5)-(6): reward magnitudes depend on the CURRENT
            # absolute error, with deliberately asymmetric negative feedback.
            heading_delta = previous_heading - heading
            lateral_delta = previous_lateral - lateral
            heading_deterioration_coeff = 6.0
            lateral_deterioration_coeff = 72.0
            if heading_delta > 1.0:
                heading_term = 0.6 * heading
            elif heading_delta < -1.0:
                heading_term = -heading_deterioration_coeff * heading
            else:
                heading_term = 0.0
            if lateral_delta > 0.05:
                lateral_term = 7.2 * lateral
            elif lateral_delta < -0.05:
                lateral_term = -lateral_deterioration_coeff * lateral
            else:
                lateral_term = 0.0

        total = float(np.clip(
            base + heading_term + lateral_term,
            self.cfg.reward_clip_min,
            self.cfg.reward_clip_max,
        ))
        return RewardOutput(total, base, heading_term, lateral_term)
