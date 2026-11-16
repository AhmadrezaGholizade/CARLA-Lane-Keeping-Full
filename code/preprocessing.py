from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SemanticFrameStack:
    """Two consecutive class-color semantic frames.

    Frames are kept as uint8 class-color images for compact replay storage. The
    network converts the resulting six channels to float and scales them to
    [0, 1] during the forward pass.
    """

    maxlen: int = 2
    _frames: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=2))

    def reset(self) -> None:
        self._frames.clear()

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected semantic frame HxWx3, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if not self._frames:
            self._frames.extend((frame.copy(), frame.copy()))
        else:
            self._frames.append(frame.copy())
        return self.stacked()

    def stacked(self) -> np.ndarray:
        if len(self._frames) != 2:
            raise RuntimeError("Frame stack has not been initialized.")
        # HWC + HWC -> 6xHxW.
        return np.concatenate(tuple(self._frames), axis=2).transpose(2, 0, 1).copy()


def normalized_speed(speed_kmh: float, max_speed_kmh: float = 110.0) -> float:
    if not np.isfinite(speed_kmh):
        raise ValueError(f"Non-finite speed: {speed_kmh}")
    # Manuscript §3.2 defines division by 110 km/h, without clipping.
    return float(speed_kmh / max_speed_kmh)


def extract_speed_kmh(observation: dict, info: dict | None = None) -> float:
    if "speed" in observation:
        return float(observation["speed"])
    if info and "speed" in info:
        return float(info["speed"])
    raise KeyError("Neither observation nor info contains a 'speed' value.")
