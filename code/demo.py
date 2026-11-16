"""Run the supplied policy with the original curved-road demo settings."""
import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from demo_env import CarEnv
from demo_settings import SETTINGS, DEMO_ACTIONS
from model import DQN


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-ip', default='127.0.0.1')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    with (root / 'data/town04_points_curve.csv').open(newline='') as file:
        rows = csv.reader(file)
        next(rows)
        starts = [int(row[0]) for row in rows]
    model = DQN(45).to(args.device).eval()
    checkpoint = torch.load(root / 'checkpoint_demo.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'], strict=True)
    env = CarEnv(dict(SETTINGS, carla_server_ip=args.server_ip))
    try:
        env.reset(starts)
        obs, _, done, _ = env.step(DEMO_ACTIONS[0])
        frames = deque(maxlen=2)
        cv2.namedWindow('Game Camera', cv2.WINDOW_NORMAL)
        with torch.inference_mode():
            while not done:
                frame = np.transpose(obs['segmentation'], (2, 0, 1)).astype(np.uint8)
                if not frames:
                    frames.append(frame)
                frames.append(frame)
                image = torch.from_numpy(np.concatenate(frames)).unsqueeze(0).to(args.device)
                speed = torch.zeros(1, device=args.device)
                action = int(model(image, speed)[0, :21].argmax().item())
                obs, _, done, _ = env.step(DEMO_ACTIONS[action])
                cv2.imshow('Game Camera', obs['game_camera'])
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
