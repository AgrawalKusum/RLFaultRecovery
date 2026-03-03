import os
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from motion_imitation.envs import env_builder
from shimmy import GymV21CompatibilityV0


class NominalWalker:
    """
    Wrapper around trained PPO model for nominal walking.
    Provides clean inference interface independent of runpard.
    """

    def __init__(
        self,
        model_path,
        stats_path,
        motion_file="motion_imitation/data/motions/dog_pace.txt",
        render=False,
    ):
        self.motion_file = motion_file
        self.render = render

        # Build environment
        self.env = SubprocVecEnv([self._make_env])

        # Load VecNormalize statistics
        if stats_path is not None and os.path.exists(stats_path):
            self.env = VecNormalize.load(stats_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
        else:
            raise RuntimeError("VecNormalize stats file not found.")

        # Load trained PPO model
        self.model = PPO.load(model_path, env=self.env)

        self.obs = None

    def _make_env(self):
        """
        Internal environment builder.
        Mirrors training-time environment.
        """
        raw_env = env_builder.build_imitation_env(
            motion_files=[self.motion_file],
            num_parallel_envs=1,
            mode="test",
            enable_randomizer=False,
            enable_rendering=self.render,
        )

        env = GymV21CompatibilityV0(env=raw_env)
        return Monitor(env)

    def reset(self):
        """
        Reset environment and return initial observation.
        """
        self.obs = self.env.reset()
        return self.obs

    def act(self, obs=None, deterministic=True):
        """
        Given observation, return action from trained policy.
        """
        if obs is None:
            obs = self.obs

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def step(self, action):
        """
        Step environment using provided action.
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        return obs, reward, done, info

    def run_episode(self, max_steps=10000):
        """
        Convenience function to run a full episode.
        """
        obs = self.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = self.act(obs)
            obs, reward, done, info = self.step(action)
            total_reward += reward[0]

            if done[0]:
                break

        return total_reward

    def close(self):
        self.env.close()