import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
file = "./julia_env"

class WL_Base:

    def __init__(self, env_module):
        self.env = env_module.init_env
        self.step_fn = env_module.step_b
        self.reset_fn = env_module.reset_b
        self.get_state_fn = env_module.get_state
    
    @classmethod
    def getActionSpace(cls, num = 1): # TODO：不对称action也需要考虑

        return spaces.Box(low=-1.0, high=1.0, shape=(num,), dtype=np.float32)
    
    @classmethod
    def getObservationSpace(cls, num = 4):

       return spaces.Box(low=-np.inf, high=np.inf, shape=(num,), dtype=np.float32)
    

class VIVEnv(WL_Base):
    def __init__(self):
        super().__init__(self.getEnv())
    
    @classmethod
    def getEnv(cls):
        file_name = os.path.join(file, "VIV_Env_3.jl")
        Main.include(file_name)
        return Main.VIVEnv
    

class FoilEnv(WL_Base):
    def __init__(self):
        super().__init__(self.getEnv())
    
    @classmethod
    def getEnv(cls):
        file_name = os.path.join(file, "foil_Env.jl")
        Main.include(file_name)
        return Main.FOILEnv
    
class DragEnv(WL_Base):
    def __init__(self):
        super().__init__(self.getEnv())
    
    @classmethod
    def getEnv(cls):
        file_name = os.path.join(file, "Drag_Env.jl")
        Main.include(file_name)
        return Main.DragEnv
    
class ShapeOpEnv(WL_Base):
    def __init__(self):
        super().__init__(self.getEnv())
    
    @classmethod
    def getEnv(cls):
        file_name = os.path.join(file, "shapeop_Env.jl")
        Main.include(file_name)
        return Main.FOILEnv
        

class JuliaEnv(gym.Env): 
    """A custom Gym environment for interacting with Julia-based simulations.
    
    This environment provides a bridge between OpenAI Gym and Julia-based 
    simulations. It handles action scaling, observation
    retrieval, and rendering.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}
        
    def __init__(self, render_mode=None, env = VIVEnv, max_episode_steps=100, 
                 statics = None, variables = None, spaces = None, verbose=False):
        # === Julia  ===
        self.wf = env()
        self.action_space = env.getActionSpace(num = spaces["action"])                          # action space
        self.observation_space = env.getObservationSpace(num = spaces["observation"])           # observation space  

        # params for constuct the scenario
        self.reset_action = 0.0
        self.scale = statics["action_scale"]
        self.statics = statics
        self.variables = variables

        self.env = self.wf.env()
        self.verbose = verbose
        if self.verbose:
            print("Julia VIV Environment initialized.")
        # === paras ===
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.reward_sum = 0
        # === img paras ===
        self.size = statics["size"]
        img_size = self.size * 100
        self.last_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)


    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state.
        
        :param seed: Random seed for reproducibility, defaults to None
        :type seed: int, optional
        :param options: Additional reset options, defaults to None
        :type options: dict, optional
        """
        super().reset(seed=seed)
        self.step_count = 0

        # reset the action
        action = self._resetAction(self.reset_action)

        self.wf.reset_fn(self.env, self.statics, self.variables)
        obs = self._getObservation()
        # warm
        _, _, _, img, _ = self._oneStep(action)
        self.last_img = img

        return obs, {}

    def step(self, action):
        """Execute one timestep in the environment.
        
        :param action: Action to take in the environment
        :type action: int or float
        :return: Tuple containing:
            - observation: Current state of the environment
            - reward: Reward received from the action
            - terminated: Whether the episode has ended
            - truncated: Whether the episode was truncated (max steps reached)
            - info: Additional information dictionary
        :rtype: tuple
        """
        self.step_count += 1

        action = self._scaleAction(action)

        # print("action: ", action)

        obs, reward, done, img, F = self._oneStep(action)
        
        if self.verbose:
            self.reward_sum += reward
            if done or self.step_count == self.max_episode_steps:
                # mean_reward = self.reward_sum / self.step_count
                print(f"Reward_Sum: {self.reward_sum:.2f}")
                self.reward_sum = 0
                print("done:", True)

        terminated = bool(done)
        truncated = self.step_count >= self.max_episode_steps

        self.last_img = img
        self.last_img  = np.expand_dims(self.last_img, axis=-1)
        info = {"info": F}

        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render the current state of the environment.
        
        Displays the last captured image using matplotlib.
        """
        print("render is on")

        plt.imshow(self.last_img)
        plt.axis("off")
        plt.pause(0.01)
        return None
    
    def _oneStep(self, action):
        """Execute one step in the underlying Julia environment.
        
        :param action: Action to take in the environment
        :type action: float
        :return: Tuple containing:
            - observation: Current state
            - reward: Immediate reward
            - done: Termination flag
            - image: Rendered image
            - informations: Additional information
        :rtype: tuple
        """
        observation, reward, done, image, informations = self.wf.step_fn(self.env, action, render = (self.render_mode == "rgb_array"))
        return observation, reward, done, image, informations
    
    def _resetAction(self, value):
        """Reset the action to a default value.
        
        :param value: Default action value
        :type value: float
        :return: Reset action value
        :rtype: float
        """

        return value
    
    def _getObservation(self):
        """Retrieve the current observation from the environment.
        
        :return: Current state observation
        :rtype: numpy.ndarray
        """

        return self.wf.get_state_fn(self.env)
    
    
    def _scaleAction(self, action):
        """Scale the normalized action to the environment's action range.
        
        :param action: Normalized action value
        :type action: float
        :return: Scaled action value
        :rtype: float
        """
        action_scale = np.array(self.scale) * np.array(action)
        
        action = np.clip(action_scale, -np.array(self.scale), np.array(self.scale))
        action = action[0]
        return action

    def close(self):
        """Clean up any resources used by the environment."""
        pass
    
if __name__ == "__main__":
    print("Testing JuliaDragEnv...")


