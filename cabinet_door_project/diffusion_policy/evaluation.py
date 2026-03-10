import logging
import os
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.inference import DiffusionPolicyInference

logger = logging.getLogger(__name__)

STATE_KEYS_ORDERED = [
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
]

STATE_DIMS = [3, 4, 3, 4, 2]


def extract_state(obs: dict) -> np.ndarray:
    parts = []
    for key in STATE_KEYS_ORDERED:
        parts.append(obs[key].flatten())
    return np.concatenate(parts).astype(np.float32)


def dataset_action_to_env_action(dataset_action: np.ndarray) -> np.ndarray:
    # HybridMobileBase composite controller: 12-dim action
    # [right(6), right_gripper(1), base(3), torso(1), base_mode(1)]
    env_action = np.zeros(12, dtype=np.float64)
    env_action[0:3] = dataset_action[5:8]    # eef_pos
    env_action[3:6] = dataset_action[8:11]   # eef_rot
    env_action[6] = -1.0 if dataset_action[11] < 0.5 else 1.0  # gripper binary
    env_action[7:10] = dataset_action[0:3]   # base_motion
    env_action[10] = dataset_action[3]       # reserve/torso
    env_action[11] = -1.0 if dataset_action[4] < 0.5 else 1.0  # base_mode flag
    return env_action


def env_action_to_dataset_action(env_action: np.ndarray) -> np.ndarray:
    dataset_action = np.zeros(12, dtype=np.float64)
    dataset_action[0:3] = env_action[7:10]   # base_motion
    dataset_action[3] = env_action[10]       # reserve/torso
    dataset_action[4] = env_action[11]       # base_mode → control_mode
    dataset_action[5:8] = env_action[0:3]    # eef_pos
    dataset_action[8:11] = env_action[3:6]   # eef_rot
    dataset_action[11] = env_action[6]       # gripper
    return dataset_action


def create_env(split: str = "pretrain", seed: int = 0):
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import create_env as _create_env
    return _create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )


def run_rollouts(
    pipeline: DiffusionPolicyInference,
    num_rollouts: int = 50,
    max_steps: int = 500,
    split: str = "pretrain",
    seed: int = 0,
    video_path: Optional[str] = None,
) -> dict:
    env = create_env(split=split, seed=seed)

    video_writer = None
    if video_path:
        import imageio
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {"successes": [], "episode_lengths": [], "rewards": []}
    n_obs_steps = pipeline.config.n_obs_steps
    n_action_steps = pipeline.config.n_action_steps

    for ep in range(num_rollouts):
        obs = env.reset()
        obs_history: deque = deque(maxlen=n_obs_steps)
        state = extract_state(obs)
        obs_history.append(state)

        ep_reward = 0.0
        success = False
        action_queue: deque = deque()

        for step in range(max_steps):
            if len(action_queue) == 0:
                while len(obs_history) < n_obs_steps:
                    obs_history.appendleft(obs_history[0])

                obs_context = torch.from_numpy(
                    np.stack(list(obs_history), axis=0)
                ).float()

                predicted_actions = pipeline.predict(obs_context)
                for i in range(predicted_actions.shape[1]):
                    action_queue.append(predicted_actions[0, i].cpu().numpy())

            dataset_action = action_queue.popleft()
            env_action = dataset_action_to_env_action(dataset_action)
            env_action = np.clip(env_action, -1.0, 1.0)

            obs, reward, done, info = env.step(env_action)
            state = extract_state(obs)
            obs_history.append(state)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(height=512, width=768, camera_name="robot0_agentview_center")[::-1]
                video_writer.append_data(frame)

            if env._check_success():
                success = True
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        logger.info(
            "Episode %d/%d: %s (steps=%d, reward=%.1f)",
            ep + 1, num_rollouts, status, step + 1, ep_reward,
        )

    if video_writer:
        video_writer.close()
    env.close()

    success_rate = sum(results["successes"]) / num_rollouts
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])
    logger.info(
        "Results: success=%.1f%% (%d/%d), avg_len=%.1f, avg_reward=%.3f",
        success_rate * 100, sum(results["successes"]), num_rollouts, avg_length, avg_reward,
    )

    return results
