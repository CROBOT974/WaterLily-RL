import os
import gymnasium as gym
import argparse
import julia_env

os.environ["JULIA_NUM_THREADS"] = "8"
from julia import Julia
jl = Julia(compiled_modules=False)

from julia import Main
print(Main.eval("Threads.nthreads()"))
from gym_base import JuliaEnv

"""

Lib支持

"""
import sys
import numpy as np
import torch.nn as nn
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import imageio
import os
# dir videos
video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)
savefile = os.path.join("./model",)

"""

反馈reward和建立checkpoint

"""
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = None
        self.step = 0

    def _on_training_start(self) -> None:
        self.current_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        self.current_rewards += rewards
        self.step += 1

        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self.current_rewards[i]/self.step)
                print("step:", self.step)
                print("current_rewards", self.current_rewards[i]/self.step)
                self.current_rewards[i] = 0.0
                self.step = 0.0
                print("reward is recorded")

        return True

checkpoint_callback = CheckpointCallback(
    save_freq= 10000,
    save_path="./checkpoints/",
    name_prefix="ppo_model",
    save_replay_buffer=True,
    save_vecnormalize=True
)

def learn(env_name = "VIV-v0", save_file = "PPO_model", total_timesteps = 80_000):
    """

    单线程环境建立，训练，保持(注册表)

    """
    env = DummyVecEnv([lambda: gym.make(env_name)])

    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=1,
        device = 'cpu'
    )
    reward_callback = RewardLoggerCallback()
    callback = CallbackList([checkpoint_callback, reward_callback])
    model.learn(total_timesteps=total_timesteps, callback = callback)
    model.save(save_file)
    rewards = np.array(reward_callback.episode_rewards)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn', action='store_true', help='play or learn.')
    parser.add_argument('--env_name', type=str, default='VIV-v0', help='environment name.')
    parser.add_argument('--filename', type=str, default='PPO_model', help='filename to save/load model.')
    parser.add_argument('--total_timesteps', type=int, default=80_000, help='total timesteps.')
    args = parser.parse_args()

    learn(args.env_name, args.filename, args.total_timesteps)