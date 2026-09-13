# agent-collaboration-sim 设计文档

> 多智能体协作仿真平台 — 基于 Boid 集群模型的起点

**版本:** 0.1.0
**日期:** 2026-09-18
**状态:** 初始设计

---

## 1. 项目目标

构建一个基于 Python 的多智能体仿真平台，以经典的 Boid 集群行为模型为起点，支持：

- 多智能体在连续空间中的自主运动模拟
- 实时提取和分析智能体间的交互网络
- 动态可视化仿真过程
- 可扩展的架构，便于后续添加新的智能体行为和协作模式

## 2. 技术栈

| 组件 | 技术选型 | 版本要求 | 用途 |
|------|----------|----------|------|
| 语言 | Python | 3.13 | 运行时 |
| 仿真框架 | Mesa | ≥3.0, <4.0 | 智能体建模、调度、数据收集 |
| 网络分析 | NetworkX | ≥3.0 | 交互网络构建与分析 |
| 可视化 | Matplotlib | ≥3.7 | 动态仿真动画、静态图表 |
| 数值计算 | NumPy + SciPy | ≥1.24 / ≥1.10 | 向量运算、空间查询 |
| 数据处理 | Pandas | ≥2.0 | 仿真数据导出与分析 |
| 测试 | Pytest | ≥7.0 | 单元测试 |
| 交互环境 | Jupyter | ≥1.0 | 探索性分析笔记本 |

## 3. 项目结构

```
agent-collaboration-sim/
├── .git/                          # 版本控制
├── .gitignore                     # 忽略规则
├── README.md                      # 项目说明
├── requirements.txt               # 依赖清单
├── setup.py                       # 开发模式安装
├── docs/
│   └── design.md                  # 本设计文档
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── boid_agent.py          # Boid 智能体定义
│   ├── models/
│   │   ├── __init__.py
│   │   └── boid_model.py          # Mesa Model 定义
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── dynamic_viz.py         # 动态可视化
│   └── utils/
│       ├── __init__.py
│       └── network.py             # 网络构建工具
├── data/                          # 仿真日志和导出数据
│   └── .gitkeep
├── notebooks/
│   ├── .gitkeep
│   └── 01_boid_exploration.ipynb  # 入门探索笔记本
├── results/                       # 可视化图表和报告
│   └── .gitkeep
└── tests/
    ├── __init__.py
    └── test_boid.py               # 单元测试
```

**设计原则：**
- `src/` 按职责分层：agents（智能体）、models（模型/调度）、visualization（可视化）、utils（工具）
- 每个模块有清晰的单一职责，通过明确接口通信
- data/ 和 results/ 不纳入版本控制的核心数据

## 4. 核心架构

### 4.1 Boid 智能体 (`src/agents/boid_agent.py`)

**类：`BoidAgent(mesa.Agent)`**

```
属性:
  - unique_id: int          — 唯一标识
  - pos: np.ndarray(2,)     — 位置向量 [x, y]
  - velocity: np.ndarray(2,) — 速度向量 [vx, vy]
  - perception_radius: float — 感知半径
  - max_speed: float        — 最大速度
  - max_force: float        — 最大力（转向限制）

方法:
  - step(model) → None
      获取感知半径内的邻居 → 计算三规则力 → 合力 → 更新速度 → 限制速度 → 移动
  - separate(neighbors) → np.ndarray(2,)
      分离力：远离过近的邻居
  - align(neighbors) → np.ndarray(2,)
      对齐力：匹配邻居的平均速度方向
  - cohere(neighbors) → np.ndarray(2,)
      聚合力：朝邻居质心移动
  - limit_vector(v, max_mag) → np.ndarray(2,)
      工具方法：限制向量最大幅值
```

**Boid 三规则实现细节：**

1. **分离 (Separation):** 对距离 < `separation_radius` 的邻居，计算排斥力与距离成反比
2. **对齐 (Alignment):** 对感知半径内的邻居，取平均速度方向作为期望方向
3. **聚合 (Cohesion):** 对感知半径内的邻居，计算质心位置，朝质心施力

合力 = `w_sep * separate + w_ali * align + w_coh * cohere`

**边界处理：** 环绕模式（toroidal）— 智能体从一边出去，从对面回来。这是 Boid 模型的标准做法，避免边界效应。

### 4.2 仿真模型 (`src/models/boid_model.py`)

**类：`BoidModel(mesa.Model)`**

```
参数:
  - n_agents: int            — 智能体数量 (默认 100)
  - width: float             — 空间宽度 (默认 200)
  - height: float            — 空间高度 (默认 200)
  - perception_radius: float — 感知半径 (默认 50)
  - separation_radius: float — 分离触发半径 (默认 25)
  - max_speed: float         — 最大速度 (默认 2.0)
  - max_force: float         — 最大力 (默认 0.3)
  - weights: tuple(float,float,float) — 三规则权重 (默认 (1.5, 1.0, 1.0))

属性:
  - space: ContinuousSpace   — Mesa 连续空间
  - network: nx.Graph        — 当前交互网络
  - datacollector: DataCollector — 数据收集器

方法:
  - __init__(params) → None
      创建空间 → 随机初始化智能体 → 配置 DataCollector
  - step() → None
      运行一个仿真步骤：所有智能体移动 → 构建交互网络 → 收集数据
  - build_network() → nx.Graph
      遍历所有智能体对，在感知半径内的创建边
  - get_agent_positions() → np.ndarray
      返回所有智能体的位置矩阵 (N, 2)
  - get_interaction_network() → nx.Graph
      返回当前交互网络的副本
```

**Mesa 3.x API 注意事项：**
- 使用 `mesa.Model` 和 `mesa.Agent`（非旧版 `mesa.time.BaseScheduler`）
- 使用 `mesa.space.ContinuousSpace` 管理连续空间
- 使用 `mesa.DataCollector` 收集仿真数据

### 4.3 网络构建工具 (`src/utils/network.py`)

**函数：`build_interaction_network(agents, radius) → nx.Graph`**
- 输入：智能体列表、感知半径
- 算法：使用 scipy.spatial.KDTree 加速近邻查询（O(n log n) 优于暴力 O(n²)）
- 输出：NetworkX 无向图，节点为 agent unique_id，边权重为距离的倒数

**函数：`network_stats(G) → dict`**
- 计算网络统计指标：节点数、边数、平均度、聚类系数、连通分量数

### 4.4 动态可视化 (`src/visualization/dynamic_viz.py`)

**函数：`run_visualization(model, steps=200, interval=50) → None`**

布局：单 Figure，两列子图
- **左图 (ax_main):** 智能体位置散点图 + 速度方向箭头 + 交互边（半透明线条）
- **右图 (ax_network):** NetworkX spring_layout 网络图，节点大小映射度数

动画控制：
- `FuncAnimation` 每帧调用 `model.step()` 并刷新两个子图
- 标题显示当前步数和网络统计信息（边数、平均度）

**函数：`save_snapshot(model, filepath) → None`**
- 保存当前帧为高清 PNG

### 4.5 数据流

```
初始化: 随机位置 + 随机速度
    ↓
每个 Step:
    1. BoidAgent.step() × N  →  每个智能体独立计算并移动
    2. model.build_network()  →  基于当前位置构建交互网络
    3. model.datacollector.collect(model)  →  记录步数据
    ↓
可视化层: 读取 model 状态 → 渲染散点 + 网络边
    ↓
导出: DataCollector → Pandas DataFrame → CSV/JSON
```

## 5. 参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| n_agents | 100 | 10-1000 | 智能体数量 |
| width / height | 200 | 50-1000 | 空间尺寸 |
| perception_radius | 50 | 10-200 | 智能体能感知邻居的距离 |
| separation_radius | 25 | 5-100 | 触发分离行为的距离阈值 |
| max_speed | 2.0 | 0.5-10 | 智能体最大移动速度 |
| max_force | 0.3 | 0.1-2.0 | 转向力的上限（控制机动性） |
| w_separation | 1.5 | 0-5 | 分离规则权重 |
| w_alignment | 1.0 | 0-5 | 对齐规则权重 |
| w_cohesion | 1.0 | 0-5 | 聚合规则权重 |

## 6. 测试策略

**单元测试 (`tests/test_boid.py`)：**
1. `test_agent_creation` — 智能体正确初始化（位置、速度、ID）
2. `test_boundary_wrapping` — 环绕边界处理正确
3. `test_separation_force` — 分离力指向正确方向
4. `test_alignment_force` — 对齐力匹配邻居平均方向
5. `test_cohesion_force` — 聚合力指向质心
6. `test_speed_limiting` — 速度不超过 max_speed
7. `test_network_construction` — 交互网络边数合理
8. `test_model_runs` — 模型能无错运行 100 步
9. `test_data_collection` — DataCollector 输出非空且格式正确

**验证标准：** `pytest tests/ -v` 全部通过

## 7. 后续扩展方向

本设计文档定义的是起点（v0.1）。后续可扩展：

- **障碍物和环境** — 在空间中添加静态障碍物
- **异构智能体** — 不同类型 Boid（捕食者/猎物）
- **通信机制** — 智能体间显式消息传递
- **涌现行为分析** — 集群分裂/合并的量化指标
- **参数扫描** — 批量实验框架
- **3D 扩展** — 从 2D 扩展到 3D 空间

这些不在当前设计范围内，但架构已为此预留扩展空间。
