"""Single-method configuration: only the paper's Full (PG-ASC+DAR+ARRS+RD-PER).

The episode distance is a fixed road-distance horizon derived from the
simulator tick budget and the manuscript's 50 km/h reference speed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROUTE_PROGRESS_VERSION = 'lane_tangent_fixed_distance_v1'


def nominal_route_distance_m(max_ticks: int, fixed_delta_seconds: float,
                             target_speed_kmh: float) -> float:
    """Nominal distance at the reference speed over the available sim time."""
    if max_ticks <= 0 or fixed_delta_seconds <= 0 or target_speed_kmh <= 0:
        raise ValueError("Ticks, simulator delta and target speed must be positive.")
    return float(max_ticks) * float(fixed_delta_seconds) * float(target_speed_kmh) / 3.6


@dataclass(frozen=True)
class EnvironmentConfig:
    carla_server_ip: str = '127.0.0.1'
    port: int = 2000
    town: str = 'Town04'
    vehicle_name: str = 'model3'
    fixed_delta_seconds: float = 0.1
    image_width: int = 400
    image_height: int = 300
    camera_fov_deg: float = 100.0
    camera_x_m: float = 1.5
    camera_z_m: float = 2.4
    traffic_vehicles: int = 0
    pedestrians: int = 0
    random_seed: int = 1
    # IDs in the supplied CSV are the only permitted episode start locations.
    spawn_csv: Path | None = None
    # Derived in train.py from max_ticks * fixed_delta * target_speed.
    route_target_distance_m: float = 2083.3333333333335
    # Reject impossible single-tick jumps so teleports cannot fake completion.
    max_progress_per_tick_m: float = 8.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1
    device: str = 'cuda'
    max_episodes: int = 6000
    max_ticks_per_episode: int = 1500
    gamma: float = 0.99
    replay_capacity: int = 30000
    replay_warmup: int = 2000
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 10.0
    target_sync_episodes: int = 20
    checkpoint_every_episodes: int = 50
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_epsilon: float = 1e-5
    priority_floor: float = 0.5
    priority_refresh_insertions: int = 256
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_episodes: int = 3000
    epsilon_power: float = 4.0
    unlock_epsilon_boost: float = 0.10
    unlock_epsilon_cap: float = 0.35
    unlock_boost_episodes: int = 100
    max_speed_normalization_kmh: float = 110.0
    target_speed_kmh: float = 50.0
    initial_speed_min_kmh: int = 10
    initial_speed_max_kmh: int = 60
    promotion_window_episodes: int = 100
    promotion_consecutive_passes: int = 3
    stage1_min_episode: int = 600
    stage2_min_age_episodes: int = 500
    stage1_completion_threshold: float = 0.60
    stage1_failure_threshold: float = 0.15
    stage1_lateral_threshold_m: float = 0.35
    stage2_completion_threshold: float = 0.75
    stage2_failure_threshold: float = 0.10
    stage2_lateral_threshold_m: float = 0.25
    collision_or_departure_penalty: float = -500.0
    stall_penalty: float = -100.0
    stall_speed_threshold_kmh: float = 3.0
    stall_ticks: int = 200
    well_aligned_lateral_m: float = 0.55
    well_aligned_heading_deg: float = 5.0
    reward_clip_min: float = -100.0
    reward_clip_max: float = 30.0
    output_root: Path = Path('outputs')
    save_replay_buffer: bool = True


def validate_config(cfg: TrainingConfig, env: EnvironmentConfig) -> None:
    if env.traffic_vehicles or env.pedestrians:
        raise ValueError('Paper protocol uses zero traffic and pedestrians.')
    if cfg.target_speed_kmh != 50.0 or cfg.replay_warmup != 2000:
        raise ValueError('Target speed / replay warm-up disagree with paper.')
    if cfg.batch_size <= 0 or cfg.replay_capacity < cfg.replay_warmup:
        raise ValueError('Invalid replay settings.')
    if cfg.max_ticks_per_episode <= 0 or cfg.max_episodes <= 0:
        raise ValueError('Invalid training horizon.')
    if env.fixed_delta_seconds <= 0 or env.max_progress_per_tick_m <= 0:
        raise ValueError('Invalid simulator tick or progress guard.')
    expected_distance = nominal_route_distance_m(
        cfg.max_ticks_per_episode, env.fixed_delta_seconds, cfg.target_speed_kmh
    )
    if abs(env.route_target_distance_m - expected_distance) > 1e-6:
        raise ValueError(
            f'Fixed route distance mismatch: expected {expected_distance:.6f} m, '
            f'got {env.route_target_distance_m:.6f} m.'
        )
