import numpy as np
import gym
from gym import spaces
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("VIV_Env_2.jl")  # module name

class JuliaVIVEnv(gym.Env): 
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, max_episode_steps=100,verbose=False):
        # === Julia  ===
        self.wf = Main.VIVEnv
        self.env = self.wf.init_env()
        self.step_fn = self.wf.step_b
        self.reset_fn = self.wf.reset_b
        self.get_state_fn = self.wf.get_state
        self.verbose = verbose
        if self.verbose:
            print("Julia VIV Environment initialized.")
        # === paras ===
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.reward_sum = 0

        # === action ===
        self.action_space = spaces.Box(low=np.array([-2], dtype=np.float32),
                                high=np.array([2.0], dtype=np.float32),
                                shape=(1,), dtype=np.float32)

        # === space ===
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3,), dtype=np.float32
        )
        self.last_img = np.zeros((400, 600, 3), dtype=np.uint8)




    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        action = np.random.uniform(1.0, 2.0)
        self.reset_fn(self.env,float(action),)
        obs = self.get_state_fn(self.env)
        # warm
        _, _, _, img, _ = self.step_fn(self.env, float(action), render=(self.render_mode == "human"))
        self.last_img = img

        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        action = action
        ax = float(action[0])  # unpack scalar
        # if self.verbose:
        #     print(f"Action: ξ = {ξ:.2f}")

        obs, reward, done, img, F = self.step_fn(
            self.env, float(ax),
            render=(self.render_mode == "human")
        )
        self.last_img = img
        self.last_img  = np.expand_dims(self.last_img, axis=-1)

        if self.verbose:
            self.reward_sum += reward
            # print(f"Step {self.step_count}/{self.max_episode_steps} — Reward: {reward:.2f}")
            if self.step_count == self.max_episode_steps:
                print(f"Reward: {self.reward_sum:.2f}")
                self.reward_sum = 0
            if done:
                print("done:",done)
                info = np.array(F)
                np.save("x_cons.npy", info)

        terminated = bool(done)
        truncated = self.step_count >= self.max_episode_steps
        info = {"img": img, "force": F}

        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            if self.last_img is None:
                return None

            fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
            canvas = FigureCanvas(fig)  # 手动设置 canvas

            ax.imshow(self.last_img, cmap="bwr", origin="lower", vmin=-5, vmax=5)
            ax.axis("off")
            fig.tight_layout(pad=0)

            canvas.draw()
            width, height = canvas.get_width_height()
            img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape((height, width, 4))[..., :3]
            plt.close(fig)
            return img

        elif self.render_mode == "human":

            plt.imshow(self.last_img)
            plt.axis("off")
            plt.pause(0.01)
        return None

    def close(self):
        pass
    
if __name__ == "__main__":
    print("Testing JuliaDragEnv...")