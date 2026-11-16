from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
import time

import numpy as np

from reward import (
    collision_from_info,
    departure_from_info,
    heading_error_from_info,
    lateral_error_from_info,
    route_completion_from_info,
)


@dataclass
class EpisodeAccumulator:
    started_at: float = field(default_factory=time.perf_counter)
    ticks: int = 0
    policy_decisions: int = 0
    q_updates: int = 0
    total_undiscounted_reward: float = 0.0
    lateral_values: list[float] = field(default_factory=list)
    heading_values: list[float] = field(default_factory=list)
    steering_values: list[float] = field(default_factory=list)
    speed_values: list[float] = field(default_factory=list)
    collision: bool = False
    lane_departure: bool = False
    route_completion: float = 0.0
    route_progress_m: float = 0.0

    def observe_tick(self, info: dict, speed_kmh: float, steer: float, reward: float) -> None:
        self.ticks += 1
        self.total_undiscounted_reward += float(reward)
        self.lateral_values.append(abs(lateral_error_from_info(info)))
        self.heading_values.append(abs(heading_error_from_info(info)))
        self.steering_values.append(float(steer))
        self.speed_values.append(float(speed_kmh))
        self.collision = self.collision or collision_from_info(info)
        self.lane_departure = self.lane_departure or departure_from_info(info)
        self.route_completion = max(self.route_completion, route_completion_from_info(info))
        self.route_progress_m = max(self.route_progress_m, float(info.get('route_progress_m', 0.0)))

    def finalize(self, action_repetition_steps: int, stage1_episode: int, stage2_episode: int) -> dict:
        lateral = float(np.mean(self.lateral_values)) if self.lateral_values else 0.0
        heading = float(np.mean(self.heading_values)) if self.heading_values else 0.0
        if len(self.steering_values) > 1:
            steering_tv = float(np.sum(np.abs(np.diff(self.steering_values))) / (len(self.steering_values) - 1))
        else:
            steering_tv = 0.0
        return {
            "route_completion": self.route_completion,
            "route_progress_m": self.route_progress_m,
            "episode_steps": self.ticks,
            "reward_per_step": self.total_undiscounted_reward / max(self.ticks, 1),
            "mean_abs_lateral_error_m": lateral,
            "mean_abs_heading_error_deg": heading,
            "steering_total_variation": steering_tv,
            "action_repetition_steps": action_repetition_steps,
            "policy_decisions": self.policy_decisions,
            "q_updates": self.q_updates,
            "wall_clock_s": time.perf_counter() - self.started_at,
            "stage1_promotion_episode": stage1_episode,
            "stage2_promotion_episode": stage2_episode,
            "collision": self.collision,
            "lane_departure": self.lane_departure,
            "average_speed_kmh": float(np.mean(self.speed_values)) if self.speed_values else 0.0,
        }


TRAINING_FIELDS = [
    "method",
    "seed",
    "episode",
    "route_completion",
    "episode_steps",
    "reward_per_step",
    "mean_abs_lateral_error_m",
    "mean_abs_heading_error_deg",
    "steering_total_variation",
    "action_repetition_steps",
    "policy_decisions",
    "q_updates",
    "wall_clock_s",
    "stage1_promotion_episode",
    "stage2_promotion_episode",
    "collision",
    "lane_departure",
    "average_speed_kmh",
    "termination_reason",
    "route_id",
    "route_length_m",
    "route_progress_m",
    "weather_condition",
    "start_spawn_index",
    "success",
    "mean_q_loss",
    "stage",
    "stage_after_episode",
    "epsilon",
    "beta",
    "rolling_completion_100",
    "elapsed_total_s",
    "first_90_episode",
    "time_to_90_wall_clock_s",
    "wheel_friction_factor",
]


class CSVLogger:
    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = fields
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, extrasaction="ignore")
        if new:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


