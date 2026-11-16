from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import os

import cv2
import numpy as np
import torch


def _encode_stack(stack: np.ndarray) -> tuple[bytes, bytes]:
    stack = np.asarray(stack, dtype=np.uint8)
    if stack.ndim != 3 or stack.shape[0] != 6:
        raise ValueError(f"Expected six-channel CHW stack, got {stack.shape}")
    first = np.ascontiguousarray(stack[:3].transpose(1, 2, 0))
    second = np.ascontiguousarray(stack[3:].transpose(1, 2, 0))
    ok1, enc1 = cv2.imencode(".png", first, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    ok2, enc2 = cv2.imencode(".png", second, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not (ok1 and ok2):
        raise RuntimeError("Could not PNG-compress semantic frame stack.")
    return enc1.tobytes(), enc2.tobytes()


def _decode_stack(encoded: tuple[bytes, bytes]) -> np.ndarray:
    frames: list[np.ndarray] = []
    for blob in encoded:
        frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Could not decode a replay-buffer semantic frame.")
        frames.append(frame)
    return np.concatenate(frames, axis=2).transpose(2, 0, 1).copy()


@dataclass
class Experience:
    state_frames: tuple[bytes, bytes]
    speed: np.float32
    action: int
    discounted_reward: np.float32
    next_frames: tuple[bytes, bytes]
    next_speed: np.float32
    done: bool
    duration: int
    action_count: int
    normalized_reward: np.float32


class PrioritizedReplayBuffer:
    """Full-method reward-deviation prioritized replay.

    Reward-aware priorities are refreshed in one vectorized pass every
    ``refresh_every`` insertions. Newly inserted transitions receive the current
    maximum priority provisionally; this raises their initial sampling chance
    without guaranteeing that they will be sampled.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        floor: float = 0.5,
        refresh_every: int = 256,
        seed: int = 1,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.floor = float(floor)
        self.refresh_every = int(refresh_every)
        self.rng = np.random.default_rng(seed)

        self.buffer: list[Experience] = []
        self.position = 0
        self.insertions_since_refresh = 0
        self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.normalized_rewards = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buffer)

    def set_beta(self, beta: float) -> None:
        self.beta = float(np.clip(beta, 0.0, 1.0))

    def push(
        self,
        state: np.ndarray,
        speed: float,
        action: int,
        discounted_reward: float,
        next_state: np.ndarray,
        next_speed: float,
        done: bool,
        duration: int,
        action_count: int,
        gamma: float,
    ) -> None:
        if duration < 1:
            raise ValueError("Transition duration must be at least one tick.")
        discount_mass = sum(gamma ** j for j in range(duration))
        normalized_reward = float(discounted_reward) / max(discount_mass, self.epsilon)
        if not np.isfinite(normalized_reward):
            raise ValueError("Non-finite duration-normalized reward.")

        experience = Experience(
            state_frames=_encode_stack(state),
            speed=np.float32(speed),
            action=int(action),
            discounted_reward=np.float32(discounted_reward),
            next_frames=_encode_stack(next_state),
            next_speed=np.float32(next_speed),
            done=bool(done),
            duration=int(duration),
            action_count=int(action_count),
            normalized_reward=np.float32(normalized_reward),
        )

        index = self.position
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[index] = experience
        self.normalized_rewards[index] = normalized_reward

        n = len(self.buffer)
        current_max = float(self.priorities[:n].max()) if n > 1 else 1.0
        self.priorities[index] = max(current_max, self.epsilon)

        self.position = (self.position + 1) % self.capacity
        self.insertions_since_refresh += 1
        if self.insertions_since_refresh >= self.refresh_every:
            self.refresh_reward_priorities()

    def refresh_reward_priorities(self) -> None:
        if not self.buffer:
            self.insertions_since_refresh = 0
            return
        n = len(self.buffer)
        values = self.normalized_rewards[:n].astype(np.float64, copy=False)
        mean = float(values.mean())
        minimum = float(values.min())
        maximum = float(values.max())
        denominator = max(abs(maximum - mean), abs(minimum - mean)) + self.epsilon
        deviation = np.abs(values - mean) / denominator
        scores = self.floor + (1.0 - self.floor) * deviation + self.epsilon
        self.priorities[:n] = np.power(scores, self.alpha).astype(np.float32)
        self.insertions_since_refresh = 0

    def sample(self, batch_size: int, device: torch.device) -> tuple[dict[str, torch.Tensor], np.ndarray, torch.Tensor]:
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError(f"Need at least {batch_size} transitions, have {n}.")

        raw = self.priorities[:n].astype(np.float64)
        total = float(raw.sum())
        if not np.isfinite(total) or total <= 0:
            probabilities = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            probabilities = raw / total

        indices = self.rng.choice(n, size=batch_size, replace=False, p=probabilities)
        experiences = [self.buffer[int(i)] for i in indices]

        states = np.stack([_decode_stack(e.state_frames) for e in experiences])
        next_states = np.stack([_decode_stack(e.next_frames) for e in experiences])
        batch = {
            "states": torch.from_numpy(states).to(device=device, dtype=torch.uint8),
            "speeds": torch.tensor([e.speed for e in experiences], device=device, dtype=torch.float32),
            "actions": torch.tensor([e.action for e in experiences], device=device, dtype=torch.long),
            "rewards": torch.tensor([e.discounted_reward for e in experiences], device=device, dtype=torch.float32),
            "next_states": torch.from_numpy(next_states).to(device=device, dtype=torch.uint8),
            "next_speeds": torch.tensor([e.next_speed for e in experiences], device=device, dtype=torch.float32),
            "dones": torch.tensor([e.done for e in experiences], device=device, dtype=torch.float32),
            "durations": torch.tensor([e.duration for e in experiences], device=device, dtype=torch.float32),
            "action_counts": torch.tensor([e.action_count for e in experiences], device=device, dtype=torch.long),
        }

        sampled_probabilities = probabilities[indices]
        weights = np.power(n * sampled_probabilities, -self.beta)
        weights /= max(float(weights.max()), self.epsilon)
        weights = weights.astype(np.float32)
        is_weights = torch.from_numpy(weights).to(device)
        return batch, indices, is_weights

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "floor": self.floor,
            "refresh_every": self.refresh_every,
            "buffer": self.buffer,
            "position": self.position,
            "insertions_since_refresh": self.insertions_since_refresh,
            "priorities": self.priorities,
            "normalized_rewards": self.normalized_rewards,
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PrioritizedReplayBuffer":
        obj = cls(
            capacity=state["capacity"],
            alpha=state["alpha"],
            beta=state["beta"],
            epsilon=state["epsilon"],
            floor=state["floor"],
            refresh_every=state["refresh_every"],
        )
        obj.buffer = state["buffer"]
        obj.position = state["position"]
        obj.insertions_since_refresh = state["insertions_since_refresh"]
        obj.priorities = state["priorities"]
        obj.normalized_rewards = state["normalized_rewards"]
        obj.rng.bit_generator.state = state["rng_state"]
        return obj

    def save(self, path: Path, episode: int | None = None) -> None:
        """Chunked, atomic serialization to limit peak CPU RAM at 30k transitions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.part')
        metadata = self.state_dict()
        frames = metadata.pop('buffer')
        metadata.update({'format': 'chunked_v2', 'count': len(frames),
                         'chunk_size': 500, 'saved_episode': episode})
        try:
            with temporary.open('wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for start in range(0, len(frames), 500):
                    pickle.dump(frames[start:start+500], handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
        finally:
            if temporary.exists(): temporary.unlink()

    @classmethod
    def load(cls, path: Path) -> 'PrioritizedReplayBuffer':
        with Path(path).open('rb') as handle:
            state = pickle.load(handle)
            if state.get('format') == 'chunked_v2':
                n = int(state.pop('count'))
                state.pop('format')
                state.pop('chunk_size')
                episode = state.pop('saved_episode', None)
                buffer = [None] * n
                pos = 0
                while pos < n:
                    chunk = pickle.load(handle)
                    buffer[pos:pos+len(chunk)] = chunk
                    pos += len(chunk)
                state['buffer'] = buffer
                obj = cls.from_state_dict(state)
                obj.saved_episode = episode
                return obj
            # Backward compatibility with previously saved complete dicts.
            return cls.from_state_dict(state)
