# agent-collaboration-sim

基于 Python + Mesa 的多智能体协作仿真平台。

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows (Git Bash):
source .venv/Scripts/activate
# Windows (CMD):
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 启动动态可视化
python -m src.visualization.dynamic_viz

# 启动 Jupyter 探索笔记本
jupyter notebook notebooks/
```

## 项目结构

```
src/
├── agents/          # 智能体定义（BoidAgent）
├── models/          # 仿真模型（BoidModel）
├── visualization/   # 动态可视化
└── utils/           # 工具函数（网络构建）
data/                # 仿真日志和导出数据
notebooks/           # Jupyter 分析笔记本
results/             # 可视化图表和报告
tests/               # 单元测试
docs/                # 设计文档
```

## Boid 模型

经典的集群行为模型，每个智能体遵循三条规则：

1. **分离** — 避开过近的邻居
2. **对齐** — 匹配邻居的移动方向
3. **聚合** — 朝邻居群体的中心移动

这些简单的局部规则产生复杂的全局集群行为。

## 技术栈

- **Mesa** — 多智能体仿真框架
- **NetworkX** — 交互网络分析
- **Matplotlib** — 动态可视化
- **NumPy / SciPy** — 数值计算与空间查询
