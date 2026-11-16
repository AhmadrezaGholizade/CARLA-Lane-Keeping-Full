"""Read-only checks before an expensive fixed-distance training run."""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path

from config import EnvironmentConfig, TrainingConfig, nominal_route_distance_m, validate_config
from env import CarlaLaneEnv


def main():
    p = argparse.ArgumentParser(description='Verify CARLA map, approved spawn IDs and fixed-distance setup.')
    p.add_argument('--server-ip', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--spawn-town04', type=Path,
                   default=Path(__file__).resolve().parent.parent/'data'/'town04_points.csv')
    p.add_argument('--max-ticks', type=int, default=1500)
    p.add_argument('--target-speed-kmh', type=float, default=50.0)
    a = p.parse_args()

    training = TrainingConfig(max_ticks_per_episode=a.max_ticks, target_speed_kmh=a.target_speed_kmh)
    base = EnvironmentConfig(town='Town04', carla_server_ip=a.server_ip, port=a.port,
                             spawn_csv=a.spawn_town04)
    distance = nominal_route_distance_m(training.max_ticks_per_episode,
                                        base.fixed_delta_seconds,
                                        training.target_speed_kmh)
    cfg = replace(base, route_target_distance_m=distance)
    validate_config(training, cfg)
    env = CarlaLaneEnv(cfg)
    try:
        actors = env.world.get_actors()
        other_vehicles = [x for x in actors.filter('vehicle.*') if getattr(x, 'is_alive', True)]
        pedestrians = [x for x in actors.filter('walker.pedestrian.*') if getattr(x, 'is_alive', True)]
        if other_vehicles or pedestrians:
            raise RuntimeError(f'Non-zero CARLA actors: {len(other_vehicles)} vehicles, {len(pedestrians)} pedestrians')
        result = {
            'status': 'PASS',
            'carla_map': env.map.name,
            'spawn_points_in_map': len(env.spawn_points),
            'allowed_start_points': len(env.allowed_starts),
            'progress_method': 'cumulative positive displacement projected onto local driving-waypoint tangent',
            'fixed_delta_seconds': base.fixed_delta_seconds,
            'max_ticks_per_episode': training.max_ticks_per_episode,
            'simulated_time_budget_s': training.max_ticks_per_episode * base.fixed_delta_seconds,
            'reference_speed_kmh': training.target_speed_kmh,
            'fixed_target_distance_m': distance,
            'success_threshold_distance_m': 0.95 * distance,
            'other_traffic_vehicles': len(other_vehicles),
            'pedestrians': len(pedestrians),
            'note': 'PASS validates the start-point and fixed-distance setup, not a historical CARLA run or paper-result reproduction.',
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == '__main__':
    main()
