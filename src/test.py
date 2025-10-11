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
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import os
from src.gif import create_GIF
# dir videos
video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)


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

def learn(env_name, save_file, total_timesteps = 80_000):
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
    model.save(os.path.join("./model",save_file))
    rewards = np.array(reward_callback.episode_rewards)
    env.close()

def play(env_name, load_file, gif_file, data_file):
    # same simulation env while 'render_mode' is 'rgb_array' to create images
    env = gym.make(env_name)

    # load the trained PPO_model
    model = PPO.load(os.path.join("./model",load_file), env=env)

    # reset the env
    # print("测试",env.reset())
    obs, _ = env.reset()

    done = False
    truncated = False

    # if 'not done', then continue to perform the simulation operation based on trained model
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    # save as gif
    input_frame = "images"
    output_gif = os.path.join("./result/gif",gif_file)
    create_GIF(input_frame, output_gif)
    env.close()

    # save the info
    np.save(os.path.join("./result/data",data_file), info["info"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='play or learn.')
    parser.add_argument('--env_name', type=str, default='VIV-v0', help='environment name.')
    parser.add_argument('--filename', type=str, default='PPO_model', help='filename to save/load model.')
    parser.add_argument('--giffilename', type=str, default='train_policy_demo.gif', help='filename to save/load gif.')
    parser.add_argument('--datafilename', type=str, default='info_PPO.npy', help='filename to save/load data.')
    parser.add_argument('--total_timesteps', type=int, default=80_000, help='total timesteps.')
    args = parser.parse_args()
    if args.play:
        print("开始测试")
        play(args.env_name, args.filename, args.giffilename, args.datafilename)
    else:
        learn(args.env_name, args.filename, args.total_timesteps)