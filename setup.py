from setuptools import setup, find_packages

setup(
    name="RL-WaterLily",  # 包名称
    version="0.1.0",          # 版本号
    description="RL-WaterLiy: a simulation framework for RL in fluid gynamics",     # 简短描述
    long_description=open("README.md").read(),  # 详细描述（通常从README读取）
    long_description_content_type="text/markdown",  # README格式
    author="Chi Cheng, Aoming Liang, Boai Sun, Yuqi Yan, Dixia Fan",
    author_email="cc610649834@163.com",
    url="https://github.com/CROBOT974/RL-WaterLily",
    python_requires=">=3.10",
    packages=find_packages(),  # 自动查找所有包
    install_requires=[         # 依赖列表
        "numpy>=2.2.6",
        "julia",
        "gymnasium",
        "stable-baselines3",
        "matplotlib",
    ],
    classifiers=[              # 分类信息（可选）
        "License :: OSI Approved :: MIT License",
    ],
)