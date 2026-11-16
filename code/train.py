"""Train ONLY the Full method from the revised CARLA lane-keeping paper.

Run from code/: python train.py --seed 1 --server-ip 127.0.0.1
For full publication-scale protocol repeat independent seeds (1..5).
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict, replace
import json
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from actions import ACTIONS
from checkpoint import load_checkpoint, save_checkpoint
from config import (EnvironmentConfig, TrainingConfig, validate_config,
                    nominal_route_distance_m, ROUTE_PROGRESS_VERSION)
from curriculum import ActionCurriculum, EpisodePerformance
from env import CarlaLaneEnv
from metrics import CSVLogger, EpisodeAccumulator, TRAINING_FIELDS
from model import DQN
from preprocessing import SemanticFrameStack, normalized_speed
from replay_buffer import PrioritizedReplayBuffer
from reward import LaneKeepingReward, collision_from_info, departure_from_info


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def epsilon_at_episode(cfg, episode):
    x = max(0.0, 1.0 - (episode - 1) / (cfg.epsilon_decay_episodes - 1))
    return cfg.epsilon_min + (cfg.epsilon_start - cfg.epsilon_min) * x ** cfg.epsilon_power


def beta_at_episode(cfg, episode):
    return cfg.priority_beta_start + (cfg.priority_beta_end - cfg.priority_beta_start) * (episode - 1) / max(cfg.max_episodes - 1, 1)


def repetition_at_episode(episode):
    return 3 if episode <= 1500 else 2 if episode <= 3000 else 1


class FullTrainer:
    def __init__(self, cfg: TrainingConfig, env_cfg: EnvironmentConfig):
        validate_config(cfg, env_cfg)
        seed_everything(cfg.seed)
        self.cfg, self.env_cfg = cfg, env_cfg
        if cfg.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested, but PyTorch reports no available GPU. Use --device cpu for a test.')
        self.device = torch.device(cfg.device)
        self.env = CarlaLaneEnv(env_cfg)
        self.policy = DQN(45).to(self.device)
        self.target = DQN(45).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = AdamW(self.policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.replay = PrioritizedReplayBuffer(
            capacity=cfg.replay_capacity, alpha=cfg.priority_alpha,
            beta=cfg.priority_beta_start, epsilon=cfg.priority_epsilon,
            floor=cfg.priority_floor, refresh_every=cfg.priority_refresh_insertions,
            seed=cfg.seed+303,
        )
        self.reward = LaneKeepingReward(cfg)
        self.curriculum = ActionCurriculum(cfg)
        self.frames = SemanticFrameStack()
        self.init_rng = np.random.default_rng(cfg.seed + 101)
        self.action_rng = np.random.default_rng(cfg.seed + 202)
        self.run_dir = cfg.output_root / f'seed_{cfg.seed}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_completion = deque(maxlen=100)
        self.first_90_episode = None
        self.first_90_elapsed_s = None
        self.started_at = time.perf_counter()
        self.previous_elapsed_s = 0.0
        self.current_episode = 0
        # Save the exact allowed spawn-point list. No start/end route pairs are
        # generated: every episode uses one approved start and the same fixed
        # road-distance horizon.
        spawn_path = Path(env_cfg.spawn_csv)
        spawn_bytes = spawn_path.read_bytes()
        spawn_sha = hashlib.sha256(spawn_bytes).hexdigest()
        spawn_manifest = self.run_dir / 'spawn_points_used.csv'
        config_path = self.run_dir / 'training_config.json'
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding='utf-8'))
            if previous.get('route_progress_version') != ROUTE_PROGRESS_VERSION:
                raise RuntimeError(
                    'This output directory was created with a different route-completion '
                    'definition. Do not resume/mix it; use a new --out directory.'
                )
            previous_training = previous.get('training', {})
            previous_environment = previous.get('environment', {})
            protocol_mismatch = (
                int(previous_training.get('max_ticks_per_episode', -1)) != cfg.max_ticks_per_episode or
                abs(float(previous_environment.get('fixed_delta_seconds', -1.0)) - env_cfg.fixed_delta_seconds) > 1e-12 or
                abs(float(previous_training.get('target_speed_kmh', -1.0)) - cfg.target_speed_kmh) > 1e-12
            )
            if (previous.get('spawn_csv_sha256') != spawn_sha or protocol_mismatch or
                abs(float(previous.get('route_target_distance_m', -1.0)) - env_cfg.route_target_distance_m) > 1e-6):
                raise RuntimeError('Existing training run has different spawn, tick-budget or fixed-distance inputs. '
                                   'Choose a new --out directory.')
        if spawn_manifest.exists() and spawn_manifest.read_bytes() != spawn_bytes:
            raise RuntimeError('Existing run uses a different spawn-point CSV. '
                               'Choose a new --out directory for a new protocol.')
        if not spawn_manifest.exists():
            spawn_manifest.write_bytes(spawn_bytes)
        config_path.write_text(json.dumps({
            'method': 'Full',
            'route_progress_version': ROUTE_PROGRESS_VERSION,
            'training': {k: (str(v) if isinstance(v, Path) else v) for k,v in asdict(cfg).items()},
            'environment': {k: (str(v) if isinstance(v, Path) else v) for k,v in asdict(env_cfg).items()},
            'spawn_csv_sha256': spawn_sha,
            'spawn_points_manifest': str(spawn_manifest),
            'route_target_distance_m': env_cfg.route_target_distance_m,
            'episode_time_budget_s': cfg.max_ticks_per_episode * env_cfg.fixed_delta_seconds,
            'route_progress_definition': 'Cumulative positive displacement projected onto the local CARLA driving-waypoint tangent; reverse/lateral motion is not counted.',
            'distance_basis': 'max_ticks_per_episode * fixed_delta_seconds * target_speed_kmh / 3.6',
            'target_speed_role': 'Reward reference speed; not a hard speed controller.',
            'spawn_scope': 'Only CSV-listed Town04 start IDs are sampled; the CSV is assumed to be curated for the intended highway loop. The code validates Driving-waypoint projection but does not infer a topological loop.',
            'carla_map': getattr(getattr(self.env, 'map', None), 'name', 'test_double'),
        }, indent=2), encoding='utf-8')
        self.logger = CSVLogger(self.run_dir / 'training_log.csv', TRAINING_FIELDS)

    def select_action(self, frame, speed, epsilon, n_actions):
        if self.action_rng.random() < epsilon:
            return int(self.action_rng.integers(n_actions))
        self.policy.eval()
        with torch.inference_mode():
            obs = torch.from_numpy(frame).unsqueeze(0).to(self.device)
            spd = torch.tensor([speed], dtype=torch.float32, device=self.device)
            q = self.policy(obs, spd)[0]
            action_id = int(q[:n_actions].argmax().item())
        self.policy.train()
        return action_id

    def learn_step(self, episode):
        self.replay.set_beta(beta_at_episode(self.cfg, episode))
        batch, _, weights = self.replay.sample(self.cfg.batch_size, self.device)
        selected = self.policy(batch['states'], batch['speeds']).gather(1, batch['actions'][:,None]).squeeze(1)
        with torch.no_grad():
            self.policy.eval()
            online = self.policy(batch['next_states'], batch['next_speeds'])
            self.policy.train()
            action_idx = torch.arange(45, device=self.device)[None,:]
            online = online.masked_fill(action_idx >= batch['action_counts'][:,None], -torch.inf)
            next_actions = online.argmax(dim=1)
            target_value = self.target(batch['next_states'], batch['next_speeds']).gather(1, next_actions[:,None]).squeeze(1)
            discount = torch.pow(torch.full_like(batch['durations'], self.cfg.gamma), batch['durations'])
            y = batch['rewards'] + (1.0 - batch['dones']) * discount * target_value
        loss_each = nn.functional.smooth_l1_loss(selected, y, reduction='none')
        loss = (weights * loss_each).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.gradient_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    def run_episode(self, episode: int):
        self.frames.reset()
        start_speed = int(self.init_rng.integers(self.cfg.initial_speed_min_kmh, self.cfg.initial_speed_max_kmh+1))
        obs, prev = self.env.reset(start_speed=start_speed, condition='Clear')
        state = self.frames.push(obs['segmentation'])
        speed = normalized_speed(obs['speed'], self.cfg.max_speed_normalization_kmh)
        acc = EpisodeAccumulator()
        stall_ticks = 0
        period = repetition_at_episode(episode)
        done = False
        reason = 'time_limit'
        loss_sum, loss_n = 0., 0
        last_info = prev

        while not done and acc.ticks < self.cfg.max_ticks_per_episode:
            n_actions = self.curriculum.action_count
            base = epsilon_at_episode(self.cfg, episode)
            eps = self.curriculum.effective_epsilon(base, episode)
            action_id = self.select_action(state, speed, eps, n_actions)
            action = ACTIONS[action_id]
            acc.policy_decisions += 1
            discounted_reward = 0.
            duration = 0
            next_state, next_speed = state, speed

            for j in range(period):
                if acc.ticks >= self.cfg.max_ticks_per_episode:
                    break
                obs, _, env_done, info = self.env.step(action.as_tuple())
                next_state = self.frames.push(obs['segmentation'])
                next_speed = normalized_speed(obs['speed'], self.cfg.max_speed_normalization_kmh)
                stall_ticks = stall_ticks + 1 if obs['speed'] < self.cfg.stall_speed_threshold_kmh else 0
                stalled = stall_ticks >= self.cfg.stall_ticks
                r = self.reward.compute(info, prev, obs['speed'], stalled=stalled)
                discounted_reward += self.cfg.gamma**j * r.total
                duration += 1
                acc.observe_tick(info, obs['speed'], action.steer, r.total)
                prev = info
                last_info = info
                failed = collision_from_info(info) or departure_from_info(info) or stalled
                success = acc.route_completion >= .95 and not failed
                timeout = acc.ticks >= self.cfg.max_ticks_per_episode
                done = bool(env_done) or failed or success or timeout
                if collision_from_info(info): reason = 'collision'
                elif departure_from_info(info): reason = 'lane_departure'
                elif stalled: reason = 'stall'
                elif success: reason = 'completed'
                elif timeout: reason = 'time_limit'
                elif env_done: reason = 'environment_done'
                if done: break

            if duration == 0:
                break
            self.replay.push(state, speed, action_id, discounted_reward, next_state,
                             next_speed, done, duration, n_actions, self.cfg.gamma)
            state, speed = next_state, next_speed
            if len(self.replay) >= self.cfg.replay_warmup:
                loss_sum += self.learn_step(episode)
                loss_n += 1
                acc.q_updates += 1
        row = acc.finalize(period, self.curriculum.stage1_promotion_episode, self.curriculum.stage2_promotion_episode)
        row.update({
            'termination_reason': reason, 'route_id': last_info.get('route_id',''),
            'route_length_m': last_info.get('route_length_m',0.),
            'route_progress_m': last_info.get('route_progress_m',0.),
            'start_spawn_index': last_info.get('start_spawn_index',''),
            'weather_condition':'Clear', 'wheel_friction_factor':1.,
            'mean_q_loss': loss_sum/max(loss_n,1),
            'success': int(reason == 'completed'),
            'stage': self.curriculum.stage, 'epsilon':eps if acc.policy_decisions else epsilon_at_episode(self.cfg,episode),
            'beta': beta_at_episode(self.cfg,episode),
        })
        return row

    def _elapsed(self):
        return self.previous_elapsed_s + (time.perf_counter() - self.started_at)

    def _checkpoint(self, episode):
        runtime = {
            'init_rng': self.init_rng.bit_generator.state,
            'action_rng': self.action_rng.bit_generator.state,
            'env_rng': self.env.get_rng_state(),
            'elapsed_s': self._elapsed(),
            'rolling_completion': list(self.rolling_completion),
            'first_90_episode': self.first_90_episode,
            'first_90_elapsed_s': self.first_90_elapsed_s,
            'route_progress_version': ROUTE_PROGRESS_VERSION,
            'route_target_distance_m': self.env_cfg.route_target_distance_m,
            'fixed_delta_seconds': self.env_cfg.fixed_delta_seconds,
            'max_ticks_per_episode': self.cfg.max_ticks_per_episode,
            'target_speed_kmh': self.cfg.target_speed_kmh,
        }
        # Historical episode snapshot: model/optimizer/curriculum, without large replay.
        save_checkpoint(self.run_dir / f'checkpoint_{episode:04d}.pth', episode, self.policy,
                        self.target, self.optimizer, self.curriculum.state_dict(), runtime_state=runtime)
        # Single resumable snapshot including replay, overwritten every 50 episodes.
        save_checkpoint(self.run_dir / 'latest.pth', episode, self.policy, self.target,
                        self.optimizer, self.curriculum.state_dict(),
                        replay_buffer=self.replay if self.cfg.save_replay_buffer else None,
                        runtime_state=runtime)

    def resume(self, path: Path):
        start, stage, replay, runtime = load_checkpoint(path, self.policy, self.target,
                                                       self.optimizer, load_replay=self.cfg.save_replay_buffer)
        if runtime.get('route_progress_version') != ROUTE_PROGRESS_VERSION:
            raise RuntimeError(
                'Checkpoint uses a different route-completion definition. '
                'Start a fresh run; do not resume across route-progress versions.'
            )
        if (abs(float(runtime.get('route_target_distance_m', -1.0)) - self.env_cfg.route_target_distance_m) > 1e-6 or
            abs(float(runtime.get('fixed_delta_seconds', -1.0)) - self.env_cfg.fixed_delta_seconds) > 1e-12 or
            int(runtime.get('max_ticks_per_episode', -1)) != self.cfg.max_ticks_per_episode or
            abs(float(runtime.get('target_speed_kmh', -1.0)) - self.cfg.target_speed_kmh) > 1e-12):
            raise RuntimeError('Checkpoint fixed-distance/tick-budget configuration differs from current run.')
        if self.cfg.save_replay_buffer and replay is None:
            raise RuntimeError('Full resume requires a replay snapshot. Use outputs/seed_N/latest.pth.')
        if replay is not None: self.replay = replay
        self.curriculum.load_state_dict(stage)
        if 'init_rng' in runtime:
            self.init_rng.bit_generator.state = runtime['init_rng']
        if 'action_rng' in runtime:
            self.action_rng.bit_generator.state = runtime['action_rng']
        if 'env_rng' in runtime:
            self.env.set_rng_state(runtime['env_rng'])
        self.previous_elapsed_s = float(runtime.get('elapsed_s',0.))
        self.started_at = time.perf_counter()
        self.rolling_completion = deque(runtime.get('rolling_completion', []), maxlen=100)
        self.first_90_episode = runtime.get('first_90_episode')
        self.first_90_elapsed_s = runtime.get('first_90_elapsed_s')
        self.target.eval()
        return start

    def run(self, start_episode=1):
        for episode in range(start_episode, self.cfg.max_episodes+1):
            row = self.run_episode(episode)
            metrics = EpisodePerformance(row['route_completion'],
                bool(row['collision'] or row['lane_departure']), row['mean_abs_lateral_error_m'])
            if self.curriculum.update(metrics, episode):
                print(f'Action curriculum promoted to stage {self.curriculum.stage} at episode {episode}')
            self.rolling_completion.append(row['route_completion'])
            rolling = float(np.mean(self.rolling_completion))
            if len(self.rolling_completion) == 100 and rolling >= .90 and self.first_90_episode is None:
                self.first_90_episode = episode
                self.first_90_elapsed_s = self._elapsed()
            row.update({'method':'Full', 'seed':self.cfg.seed, 'episode':episode,
                        'stage_after_episode':self.curriculum.stage,
                        'stage1_promotion_episode':self.curriculum.stage1_promotion_episode,
                        'stage2_promotion_episode':self.curriculum.stage2_promotion_episode,
                        'rolling_completion_100':rolling,
                        'elapsed_total_s':self._elapsed(),
                        'first_90_episode':self.first_90_episode if self.first_90_episode is not None else '',
                        'time_to_90_wall_clock_s':self.first_90_elapsed_s if self.first_90_elapsed_s is not None else ''})
            self.logger.write(row)
            if episode % self.cfg.target_sync_episodes == 0:
                self.target.load_state_dict(self.policy.state_dict())
                self.target.eval()
            if episode % self.cfg.checkpoint_every_episodes == 0 or episode == self.cfg.max_episodes:
                self._checkpoint(episode)
            if episode == 1 or episode % 10 == 0:
                print(f'Full seed={self.cfg.seed} ep={episode}/{self.cfg.max_episodes} '
                      f'k={repetition_at_episode(episode)} actions={self.curriculum.action_count} '
                      f'completion={row["route_completion"]:.3f} rolling100={rolling:.3f} '
                      f'loss={row["mean_q_loss"]:.3f}', flush=True)
        self.logger.close()

    def close(self):
        self.logger.close()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description='Paper Full method (no ablations).')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--server-ip', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--spawn-csv', type=Path, default=Path(__file__).resolve().parent.parent/'data'/'town04_points.csv')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent.parent/'outputs')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--episodes', type=int, default=6000, help='Default article budget; shorter runs only for smoke tests')
    parser.add_argument('--no-replay-save', action='store_true', help='No true replay resume possible')
    args = parser.parse_args()
    cfg = TrainingConfig(seed=args.seed, device=args.device, max_episodes=args.episodes,
                         output_root=args.out, save_replay_buffer=not args.no_replay_save)
    base_env_cfg = EnvironmentConfig(carla_server_ip=args.server_ip, port=args.port,
                    town='Town04', spawn_csv=args.spawn_csv, random_seed=args.seed)
    fixed_distance = nominal_route_distance_m(
        cfg.max_ticks_per_episode, base_env_cfg.fixed_delta_seconds, cfg.target_speed_kmh
    )
    env_cfg = replace(base_env_cfg, route_target_distance_m=fixed_distance)
    existing_log = args.out / f'seed_{args.seed}' / 'training_log.csv'
    if args.resume is None and existing_log.exists() and len(existing_log.read_text().splitlines()) > 1:
        parser.error(f'{existing_log} already contains training episodes. '                      'Pass --resume with the matching latest.pth or choose a new --out.')
    trainer = FullTrainer(cfg, env_cfg)
    try:
        trainer.run(trainer.resume(args.resume) if args.resume else 1)
    finally:
        trainer.close()


if __name__ == '__main__': main()
