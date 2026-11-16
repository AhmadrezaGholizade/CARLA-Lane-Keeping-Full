from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict

import numpy as np

from actions import admissible_action_count
from config import TrainingConfig


@dataclass
class EpisodePerformance:
    route_completion: float
    failed: bool
    mean_abs_lateral_error_m: float


class ActionCurriculum:
    """Performance-gated 21 -> 27 -> 45 action curriculum.

    A trailing 100-episode window is evaluated after each episode once enough
    history is available. Promotion requires three consecutive passing window
    evaluations, which permits promotion at arbitrary episode indices.
    Newly executed runs log the promotion episode together with the failure
    fields needed to audit the gate.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.stage = 1
        self.history: deque[EpisodePerformance] = deque(maxlen=cfg.promotion_window_episodes)
        self.consecutive_passes = 0
        self.stage1_promotion_episode = -1
        self.stage2_promotion_episode = -1
        self.stage_started_episode = 1
        self.epsilon_boost_until_episode = 0

    @property
    def action_count(self) -> int:
        return admissible_action_count(self.stage)

    def effective_epsilon(self, base_epsilon: float, episode: int) -> float:
        if episode <= self.epsilon_boost_until_episode:
            return min(self.cfg.unlock_epsilon_cap, base_epsilon + self.cfg.unlock_epsilon_boost)
        return base_epsilon

    def update(self, metrics: EpisodePerformance, episode: int) -> bool:
        if self.stage >= 3:
            return False
        self.history.append(metrics)
        if len(self.history) < self.cfg.promotion_window_episodes:
            return False

        completion = float(np.mean([m.route_completion for m in self.history]))
        failure_rate = float(np.mean([m.failed for m in self.history]))
        lateral = float(np.mean([m.mean_abs_lateral_error_m for m in self.history]))

        if self.stage == 1:
            eligible = episode >= self.cfg.stage1_min_episode
            passed = (
                eligible
                and completion >= self.cfg.stage1_completion_threshold
                and failure_rate <= self.cfg.stage1_failure_threshold
                and lateral <= self.cfg.stage1_lateral_threshold_m
            )
        else:
            eligible = episode - self.stage_started_episode + 1 >= self.cfg.stage2_min_age_episodes
            passed = (
                eligible
                and completion >= self.cfg.stage2_completion_threshold
                and failure_rate <= self.cfg.stage2_failure_threshold
                and lateral <= self.cfg.stage2_lateral_threshold_m
            )

        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        if self.consecutive_passes < self.cfg.promotion_consecutive_passes:
            return False

        self.stage += 1
        if self.stage == 2:
            self.stage1_promotion_episode = episode
        else:
            self.stage2_promotion_episode = episode
        self.stage_started_episode = episode + 1
        self.consecutive_passes = 0
        self.epsilon_boost_until_episode = episode + self.cfg.unlock_boost_episodes
        return True

    def state_dict(self) -> dict:
        return {
            "stage": self.stage,
            "history": [asdict(x) for x in self.history],
            "consecutive_passes": self.consecutive_passes,
            "stage1_promotion_episode": self.stage1_promotion_episode,
            "stage2_promotion_episode": self.stage2_promotion_episode,
            "stage_started_episode": self.stage_started_episode,
            "epsilon_boost_until_episode": self.epsilon_boost_until_episode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.stage = int(state["stage"])
        self.history.clear()
        self.history.extend(EpisodePerformance(**x) for x in state.get("history", []))
        self.consecutive_passes = int(state.get("consecutive_passes", 0))
        self.stage1_promotion_episode = int(state.get("stage1_promotion_episode", -1))
        self.stage2_promotion_episode = int(state.get("stage2_promotion_episode", -1))
        self.stage_started_episode = int(state.get("stage_started_episode", 1))
        self.epsilon_boost_until_episode = int(state.get("epsilon_boost_until_episode", 0))
