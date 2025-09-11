# 🌊Deep Reinforcement Learning with WaterLily
This project contains a simulation framework for leveraging deep reinforcement learning in studying of the fluid gynamics. Stable BaseLine 3 and WaterLily are essential.
本项目将 流体动力学仿真 (CFD) 与 强化学习 (RL) 结合，允许智能体在物理真实的流体环境中学习控制策略。

Stable-Baselines3 (SB3)：RL 算法 (PPO, SAC...)

WaterLily.jl：高性能流体动力学引擎 (Julia 实现)

Gymnasium API：封装 Julia 环境为 Python RL 环境

典型应用包括：

涡激振动 (VIV) 控制

翼型 (foil) 优化

流场控制

## Install
* **Download**
```
git clone https://github.com/CROBOT974/RL-WaterLily.git
cd RL-WaterLily
```
* **Python packages**
```
pip install -e .
```
* **Julia packages**
```
Julia
using Pkg
Pkg.add([
    "ImageCore",
    "WaterLily",
    "StaticArrays",
    "PyPlot",
    "PyCall",
    "ColorSchemes",
    "ImageIO",
    "FileIO",
    "Colors",
    "Images",
    "Statistics"
])
```
