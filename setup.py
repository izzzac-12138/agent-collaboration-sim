from setuptools import setup, find_packages

setup(
    name="agent-collaboration-sim",
    version="0.1.0",
    description="多智能体协作仿真平台 — 基于 Mesa 的 Boid 集群模拟",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "mesa>=3.0,<4.0",
        "networkx>=3.0",
        "matplotlib>=3.7",
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "jupyter>=1.0"],
    },
)
