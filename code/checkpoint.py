from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch

from actions import ACTIONS
from replay_buffer import PrioritizedReplayBuffer


def save_checkpoint(
    path: Path,
    episode: int,
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curriculum_state: dict,
    replay_buffer: PrioritizedReplayBuffer | None = None,
    runtime_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode": int(episode),
        "action_order": [action.as_tuple() for action in ACTIONS],
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "curriculum_state": curriculum_state,
        "runtime_state": runtime_state or {},
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temp = path.with_name(path.name + '.part')
    try:
        # Store the replay before replacing model checkpoint. On interruption
        # a mismatched episode ID is detected explicitly upon resume.
        if replay_buffer is not None:
            replay_buffer.save(path.with_suffix('.replay.pkl'), episode=episode)
        with temp.open('wb') as file:
            torch.save(payload, file)
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def load_checkpoint(
    path: Path,
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    load_replay: bool = True,
) -> tuple[int, dict, PrioritizedReplayBuffer | None, dict]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    action_order = [action.as_tuple() for action in ACTIONS]
    if payload.get("action_order", action_order) != action_order:
        raise ValueError("Checkpoint action order differs from this trainer.")
    policy_net.load_state_dict(payload["policy_state_dict"])
    if target_net is not None:
        target_net.load_state_dict(payload.get("target_state_dict", payload["policy_state_dict"]))
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "numpy_random_state" in payload:
        np.random.set_state(payload["numpy_random_state"])
    if "torch_random_state" in payload:
        torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])

    replay = None
    replay_path = path.with_suffix(".replay.pkl")
    if load_replay and replay_path.exists():
        replay = PrioritizedReplayBuffer.load(replay_path)
        replay_episode = getattr(replay, 'saved_episode', None)
        if replay_episode is not None and replay_episode != int(payload['episode']):
            raise RuntimeError('Replay snapshot and model checkpoint episode IDs differ. '
                               'Do not resume from mismatched files.')
    return (
        int(payload["episode"]) + 1,
        payload.get("curriculum_state", {}),
        replay,
        payload.get("runtime_state", {}),
    )
