# 🌊Deep Reinforcement Learning with WaterLily
Waterlily-RL 是一个基于 WaterLily.jl 流体动力学引擎与 Stable-Baselines3 强化学习库构建的跨语言仿真框架，主要用于流体力学场景下的智能控制与策略学习。

Waterlily-RL 为您提供了：

* 基于 Julia WaterLily 的高性能流体模拟，直接调用原生 API，无需额外 CFD 库，即可获得真实物理场

* Python 端 Gymnasium 风格环境封装，支持 Stable-Baselines3 的常用算法 (PPO, SAC, DDPG 等)

* 提供 VIV (涡激振动)、翼型控制、阻力优化 等环境示例，支持扩展新的流体力学任务

* 支持训练过程中的 渲染与可视化，方便调试与展示仿真效果

* 模块化的接口设计，便于自由组合 场景参数、动力学模型与 RL 算法

请查看文档以获取更多信息 (更新中)

Waterlily-RL is a cross-language simulation framework built on the WaterLily.jl fluid dynamics engine and the Stable-Baselines3 reinforcement learning library, mainly used for intelligent control and policy learning in fluid mechanics scenarios. 

Waterlily-RL offers you: 

* High-performance fluid simulation based on Julia WaterLily, directly calling the native API, without the need for additional CFD libraries, to obtain real fluid fields. 
* Gymnasium-style wrapper of enviroment in Python, supporting common algorithms of Stable-Baselines3 (PPO, SAC, DDPG, etc.) 
* Provide environmental examples such as VIV (Vortex-Induced Vibration), airfoil control, drag optimization, etc., and support the expansion of new fluid dynamic tasks. 
* Support rendering and visualization during the training process to facilitate debugging and demonstrating simulation effects. 
* The modular interface design facilitates the free combination of scene parameters, dynamic models, and RL algorithms. 

Please refer to the document for more information (under update).

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
pip install -r requirements.txt
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
### Run a demo(learning)
```
python test.py --env_name="VIV-v0" --total_timesteps=80_000 --filename="PPO_model" -learn
```
## Result
### Gif of the VIV scenario

![VIV-v0](./viv_simulation_2.gif)
