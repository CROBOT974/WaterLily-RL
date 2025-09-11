# 🌊Deep Reinforcement Learning with WaterLily
This project contains a simulation framework for leveraging deep reinforcement learning in studying of the fluid gynamics. Stable BaseLine 3 and WaterLily are essential.
Waterlily-RL 是一个基于 WaterLily.jl 流体动力学引擎与 Stable-Baselines3 强化学习库构建的跨语言仿真框架，主要用于流体力学场景下的智能控制与策略学习。

Waterlily-RL 为您提供了：

* 基于 Julia WaterLily 的高性能流体模拟，直接调用原生 API，无需额外 CFD 库，即可获得真实物理场

* Python 端 Gymnasium 风格环境封装，支持 Stable-Baselines3 的常用算法 (PPO, SAC, DDPG 等)

* 提供 VIV (涡激振动)、翼型控制、阻力优化 等环境示例，支持扩展新的流体力学任务

* 支持训练过程中的 渲染与可视化，方便调试与展示仿真效果

* 模块化的接口设计，便于自由组合 场景参数、动力学模型与 RL 算法

请查看文档以获取更多信息 (更新中)

## Get Started
### Preparation
* **Windows / Linux**
* **Python 3.10 +**
* **Julia 1.10.9**
### Download
```
git clone https://github.com/CROBOT974/RL-WaterLily.git
cd RL-WaterLily
```
### Python packages
```
pip install -e .
```
### Julia packages
```
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
