import gymnasium as gym
from gymnasium.envs.registration import register
from ..src.gym_base import VIVEnv, FoilEnv, DragEnv, ShapeOpEnv, JuliaEnv
import numpy as np

diameter = 64
def pos_generator():
    return [0.0, np.random.uniform(- diameter/6, diameter/6)]
def ksi_generator():
    return np.random.uniform(3.0, 4.0)


register(
    id="VIV-v0",
    entry_point = JuliaEnv,
    kwargs={
        "render_mode": "rgb_array",
        "max_episode_steps": 2000,
        "env": VIVEnv,
        "statics": {"L_unit": diameter, 
                    "F_scale": 500, 
                    "size": [10, 8], 
                    "location": [3, 4]},
        "variables": {"position":[0.0, -1.0], 
                      "velocity":[0.0, 0.0]},
        "spaces": {"action": 1, 
                   "observation": 3},
        "verbose": True
    }
)

register(
    id="Foil-v0",
    entry_point = JuliaEnv,
    kwargs={
        "render_mode": "rgb_array",
        "max_episode_steps": 2000,
        "env": FoilEnv,
        "statics": {"L_unit": diameter,
                    "F_scale": 10,
                    "size": [6, 6],
                    "nose": [1, 3],
                    "rot_center":[0.25,0.0]},
        "variables": {"position":[0.0, 0.0],
                      "velocity":[0.0, 0.0],
                      "theta": 0.95 * np.pi,
                      "rot_vel": 0.0,
                      "rot_acc": 0.0},
        "spaces": {"action": 1, 
                   "observation": 5},
        "verbose": True
    }
)

register(
    id="Foil-v0",
    entry_point = JuliaEnv,
    kwargs={
        "render_mode": "rgb_array",
        "max_episode_steps": 2000,
        "env": FoilEnv,
        "statics": {"L_unit": diameter,
                    "F_scale": 10,
                    "size": [6, 6],
                    "nose": [1, 3],
                    "rot_center":[0.25,0.0]},
        "variables": {"position":[0.0, 0.0],
                      "velocity":[0.0, 0.0],
                      "theta": 0.95 * np.pi,
                      "rot_vel": 0.0,
                      "rot_acc": 0.0},
        "spaces": {"action": 1, 
                   "observation": 5},
        "verbose": True
    }
)

# ! diameter = 48

register(
    id="Drag-v0",
    entry_point = JuliaEnv,
    kwargs={
        "render_mode": None,
        "max_episode_steps": 2000,
        "env": DragEnv,
        "statics": {"L_unit": 48,
                    "F_scale": 8,
                    "L_ratio": 0.15,
                    "L_gap":0.05,
                    "location": [2, 0],
                    "size": [6, 2]},
        "variables": {"ksi": ksi_generator},
        "spaces": {"action": 1, 
                   "observation": 2},
        "verbose": True
    }
)