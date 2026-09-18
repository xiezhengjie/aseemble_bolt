# UR5e 螺栓装配 — MuJoCo 仿真与模仿学习

基于 **MuJoCo** 的 UR5e 机械臂螺栓装配任务，采用 **SAC → BC → GAIL/GASILfD** 的三阶段模仿学习流水线。
详细技术细节见 [`docs/SAC_GAIL_完整训练报告.md`](docs/SAC_GAIL_完整训练报告.md)。

---

## 目录结构

```
ur5e_assemble_mujoco/
├── training/                 # 全部 Python 代码
│   ├── envs/                 # MuJoCo 环境
│   │   └── assemble_mujoco_env.py      # 主环境（MuJoCo 原生，**不依赖 ROS**）
│   ├── train/                # 训练入口：train_sac*.py / train_bc*.py / train_sac_gail*.py
│   ├── scripts/              # 数据采集 / 标定 / 工具脚本
│   ├── algorithm/            # RL 算法（sac / residual_sac / discriminator）+ 控制（controllers / ur5e_ik / filter / force_calibration）
│   ├── utils/                # rl_utils / math_utils / checkpoint
│   ├── experiments/          # 实验与调试脚本（**不是单元测试**）
│   └── notebooks/            # 探索性 notebook
├── docs/                     # 文档（训练报告）
├── mjcf/                     # MuJoCo 场景与物体 XML
├── urdf/                     # URDF 模型
├── assets/                   # 3D 网格与贴图（meshes 66M + textures 22M）
├── datasets/                 # 专家数据（.npz）
├── models/                   # 训练检查点
├── results/                  # 分析输出（如 CTC 带宽）
├── logs/                     # TensorBoard 日志
└── requirements.txt
```

> `experiments/` 与 `notebooks/` 由原先散落在 `envs/`、`scripts/`、`train/`、`algorithm/` 下的
> **6 个 `test*` 文件改名归位**而来 —— 它们都不是单元测试，占用 `test` 命名会被 pytest 误收集。
> 项目目前**尚无真正的单元测试**。

> `datasets/` / `models/` / `logs/` 的位置由 `training/utils/rl_utils.py` 的
> `find_project_root()` 决定 —— 它从自身向上查找 `requirements.txt`，因此**与当前工作目录无关**，
> 在任何目录下运行脚本，输出都会落回项目内。

---

## 环境安装

### 1. PyTorch（必须按 CUDA 版本单独装）

```bash
# 远程环境 CUDA 驱动 590.44 → 建议 cu124
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# 若 nvcc 是 12.1，把 cu124 改成 cu121
```

### 2. 其余依赖

```bash
pip install -r requirements.txt
```

### 3. TRAC-IK 的系统级依赖（`pytracik` 需要）

```bash
sudo apt-get update
sudo apt-get install -y libboost-all-dev libeigen3-dev liborocos-kdl-dev libnlopt-dev libnlopt-cxx-dev
```

若改用 ROS 原生版本（`trac_ik_python`）：

```bash
sudo apt install ros-$ROS_DISTRO-trac-ik
```

---

## 训练流水线

| 阶段 | 做什么 | 入口 |
|---|---|---|
| **Stage 1** | SAC 训练，生成专家数据 | `training/train/train_sac.py`、`train_sac_sb3.py`（SB3 基线对照） |
| **Stage 2** | BC 行为克隆（在专家数据上做监督学习） | `training/train/train_bc_ur5e.py` |
| **Stage 3** | GAIL / GASILfD 模仿学习 | `training/train/train_sac_gail_ur5e.py`（主脚本）、`train_sac_gail.py`（早期版本） |

### 数据采集

```bash
training/scripts/record_demonstration_data.py   # 遥操作采集专家轨迹
training/scripts/teleop_mujoco.py               # 遥操作控制
training/scripts/normalize_expert_data.py --input <原始.npz> --output <归一化.npz> [--force]
```

### 其他工具

```bash
training/scripts/camera_calibration.py          # 相机标定
training/scripts/check_expert_state_machine.py  # 专家状态机检查
training/scripts/ctc_bandwidth_qt.py            # CTC 带宽（Qt 图形界面）
training/scripts/verify_ctc_bandwidth.py        # CTC 带宽验证
```

---

## 运行方式

脚本通过 `sys.path.append(...)` 把 `training/` 加入模块搜索路径，因此**在项目根目录直接运行即可**：

```bash
cd ur5e_assemble_mujoco
python training/train/train_bc_ur5e.py
python training/train/train_sac_gail_ur5e.py            # 支持 --seed
```

### 主要输入输出

| 路径 | 说明 |
|---|---|
| `datasets/recorded_data.npz` | 原始专家数据 |
| `datasets/recorded_data_norm.npz` | 归一化后的专家数据（GAIL 训练实际使用） |
| `models/bc_model_ur5e/` | BC 检查点（GAIL 阶段的专家模型） |
| `models/sac_gail_ur5e_model/` | GAIL 训练产物 |
| `logs/sac_gail_ur5e_log/` | TensorBoard 日志 |

```bash
tensorboard --logdir logs/sac_gail_ur5e_log
```

---

## 变更记录与待办

### 已处理（2026-09-18）

1. **15 处硬编码绝对路径** → 全部改为基于 `Path(__file__).resolve().parents[2]`（项目根 `PROJECT_ROOT`）的相对路径。
   涉及 `envs/assemble_mujoco_env.py`、`envs/assemble_mujoco_moveit_env.py`、`scripts/teleop_mujoco.py`、
   `scripts/ctc_bandwidth_qt.py`、`scripts/verify_ctc_bandwidth.py`、`algorithm/ur5e_ik.py`、`algorithm/test.py`；
   文档死链改为相对链接，notebook 改为按 `requirements.txt` 标记定位项目根。
   实测 `AssembleMuJoCoEnv()` 正常加载场景（`observation_space=(10,)`、`action_space=(6,)`、`reset()` 通过）。
2. **删除全部 ROS 2 依赖**：`package.xml` / `setup.py` / `setup.cfg` / `resource/` / `test/`（ament 构建与 lint 模板）、
   `training/launch/`，以及唯一依赖 `rclpy`/`moveit` 的 `training/envs/assemble_mujoco_moveit_env.py`。
   代码中 ROS 2 引用已清零，回归验证通过。
3. **去重**：删除 `../ur5e_assemble_ws/src/training_2/`（与本项目逐字节重复，释放 321M）。
   原先只存在于那里的 `LICENSE`(Apache-2.0) 等文件已先行并入本仓库。
4. **清空壳与错位**：删 `training/deploy/`、`training/envs/assemble_real_env.py`(0 字节)；
   `training/docs/` → 项目根 `docs/`。
5. **6 个 `test*` 改名归位** → `training/experiments/`（4 个脚本）与 `training/notebooks/`（2 个 notebook）。

### 仍待处理

6. ⚠️ **运行环境**：需用**系统 python3.10**（`/usr/bin/python3.10`）—— `pinocchio` 4.0.0 装在那里；
   anaconda 的 python3.13 会报 `No module named 'pinocchio'`。
7. `urdf/ur5e_assemble.urdf` 第 729 行仍有 `file://` 绝对 mesh 引用（可视网格；IK 不解析几何，无功能影响；
   项目内无对应 dae）。如需渲染该 URDF，建议改为 `package://realsense2_description/meshes/d435.dae`。
8. `training/algorithm/` 混装 RL 算法与控制/物理代码，建议拆为 `algos/` + `control/`。
9. 超参仍硬编码在脚本顶部（38 个 `.py` 只有 4 个用 argparse），建议外置到 `configs/*.yaml`。
10. 单文件过大：`ctc_bandwidth_qt.py` 1882 行、`verify_ctc_bandwidth.py` 1861 行、`utils/rl_utils.py` 53KB（杂物抽屉）。
11. 项目**尚无真正的单元测试**；`logs/` 190M + `models/` 23M 等产物建议按实验清理。
12. 项目**尚未纳入 git**；`.gitignore` 已就绪（排除 `logs/`、`models/`、`datasets/`、缓存等）。

---

## 文档

- [SAC + GAIL 模仿学习完整训练报告](docs/SAC_GAIL_完整训练报告.md) —— 训练配置、消融实验、GASILfD 理论推导
