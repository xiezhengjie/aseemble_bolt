import json
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from gymnasium import Wrapper
from gymnasium.wrappers import FrameStackObservation
from training.envs.assemble_mujoco_env import AssembleMuJoCoEnv

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def residual_target_entropy(action_dim, residual_scale):
    """按残差动作测度修正后的 SAC 目标熵。"""
    if float(residual_scale) <= 0.0:
        raise ValueError("residual_scale 必须 > 0")
    return float(action_dim) * (np.log(float(residual_scale)) - 1.0)

def as_sequence(x, seq_len: int, raw_obs_dim: int) -> torch.Tensor:
    """把时序观测整理为 (B,T,D)。"""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    t, d = int(seq_len), int(raw_obs_dim)
    if t < 1 or d < 1:
        raise ValueError("seq_len 和 raw_obs_dim 必须为正数")
    if x.dim() == 3:
        if tuple(x.shape[1:]) != (t, d):
            raise ValueError(f"期望 (B,{t},{d})，实际 {tuple(x.shape)}")
        return x
    if x.dim() == 2:
        if tuple(x.shape) == (t, d):
            return x.unsqueeze(0)
        if x.shape[-1] == t * d:
            return x.reshape(-1, t, d)
        if t == 1 and x.shape[-1] == d:
            return x.unsqueeze(1)
    if x.dim() == 1:
        if x.numel() == t * d:
            return x.reshape(1, t, d)
        if t == 1 and x.numel() == d:
            return x.reshape(1, 1, d)
    raise ValueError(f"时序输入维度错误: shape={tuple(x.shape)}")

def orthogonal_init(module, gain=np.sqrt(2), bias=0.0):
    """
    正交初始化，支持三种传入方式：
      1. 单个 nn.Linear            -> 直接初始化
      2. nn.Sequential / 任意容器模块 -> 递归初始化内部所有 Linear
      3. 其他类型                  -> 显式报错，防止静默跳过
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, bias)
    elif isinstance(module, torch.nn.Module):
        for child in module.children():
            orthogonal_init(child, gain=gain, bias=bias)
    else:
        raise TypeError(f"orthogonal_init 不支持的类型: {type(module)}")

def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )  # Use relu for Swish
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)

    return layer

def find_project_root():
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").is_file():
            return parent
    raise FileNotFoundError("项目根目录缺少 requirements.txt")

def wrap_frame_stack(env, frame_stack, padding_type="reset"):
    """把单帧观测叠成 (T, *obs_shape)，给 GRU 用。

    episode 开始时默认重复首帧（``padding_type='reset'``）。
    """
    n = int(frame_stack)
    if n < 1:
        raise ValueError(f"frame_stack 必须 >= 1，当前: {frame_stack}")
    if n == 1:
        return env
    return FrameStackObservation(env, stack_size=n, padding_type=padding_type)


def make_env(xml_path: str, urdf_path: str, max_episode_steps: int):
    """返回一个无参 callable，调用后创建一个包装好的 env。"""
    def _init():
        env = AssembleMuJoCoEnv(
            xml_path=xml_path,
            urdf_path=urdf_path,
            render_mode=None,
            max_episodic_steps=max_episode_steps,
        )
        return EpisodeStatsWrapper(env)
    return _init

class EpisodeStatsWrapper(Wrapper):
    """回合结束时注入 info['final_info'] = {episodic_return, episodic_length, success, …}。

    适配 gym 标准环境（如 MountainCarContinuous-v0）与自定义环境（如 AssembleMuJoCoEnv）。
    success 优先取 env 的 info['success']；环境未提供该字段时不写入。
    插装任务额外转发 depth / position_error_xy / yaw_error / angle_z / state，供 TensorBoard tasks/*。
    """

    _TASK_KEYS = ("depth", "position_error_xy", "yaw_error", "angle_z", "state")

    def __init__(self, env):
        super().__init__(env)
        self._ep_return = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        self._ep_return = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_return += float(reward)
        self._ep_length += 1
        if terminated or truncated:
            # 复制 info 避免修改 env 内部 dict
            info = dict(info)
            final_info = {
                'episodic_return': float(self._ep_return),
                'episodic_length': int(self._ep_length),
            }
            if 'success' in info:
                final_info['success'] = bool(info['success'])
            for k in self._TASK_KEYS:
                if k in info:
                    value = np.asarray(info[k])
                    if value.size:
                        final_info[k] = float(value.reshape(-1)[0])
            info['final_info'] = final_info
        return obs, reward, terminated, truncated, info

class RunningMeanStd:
    """使用 Welford 在线算法维护累计均值和方差"""

    def __init__(self, shape, epsilon=1e-4, clip=50.0):
        """
        :param shape: 单个观测的 shape，如 (obs_dim,) 或 (C, H, W)
        :param epsilon: 防止除零的小值
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # 用 epsilon 初始化，避免早期除零
        self.epsilon = epsilon
        self.clip = clip
        self._torch_stats = {}

    def update(self, x):
        """用新数据更新统计量"""
        # 展平到 (N, *shape)，统一处理
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.shape == self.mean.shape:
            x = x.reshape((1,) + self.mean.shape)
        elif self.mean.ndim == 0:
            x = x.reshape(-1)
        elif x.ndim > self.mean.ndim and x.shape[-self.mean.ndim:] == self.mean.shape:
            x = x.reshape((-1,) + self.mean.shape)
        else:
            raise ValueError(f"统计输入 shape={x.shape} 与观测 shape={self.mean.shape} 不匹配")
        if x.shape[0] == 0:
            return
        self._torch_stats.clear()

        # 按第一个维度（batch/env 维度）计算
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        # Welford 在线更新
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        # 合并方差: M2 = count * var
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        """
        归一化输入，并做截断。
        :param x: ndarray 或 torch.Tensor
        :return: 同类型的归一化结果
        """
        if isinstance(x, torch.Tensor):
            if not x.is_floating_point():
                x = x.float()
            key = (x.device, x.dtype)
            stats = self._torch_stats.get(key)
            if stats is None:
                mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
                var = torch.as_tensor(self.var, dtype=x.dtype, device=x.device)
                self._torch_stats[key] = (mean, var)
            else:
                mean, var = stats
            return torch.clamp((x - mean) / torch.sqrt(var + self.epsilon), -self.clip, self.clip)

        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized.astype(np.float32)

    def state_dict(self):
        """保存状态，用于恢复训练"""
        return {
            "mean": self.mean.copy().tolist(),
            "var": self.var.copy().tolist(),
            "count": self.count,
        }

    def load_state_dict(self, state):
        """加载保存的统计量。"""
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])
        self._torch_stats = {}

    def save_normalizer(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        np.savez(_model_dir/"obs_normalizer.npz", **self.state_dict())

    def load_normalizer(self, model_dir):
        with np.load(Path(model_dir) / "obs_normalizer.npz") as data:
            self.load_state_dict(data)

class RewardNormalizer:
    """Scale immediate rewards without centering; update only on collection."""

    def __init__(self, clip=5.0):
        self.clip = float(clip)
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("reward clip must be a positive finite number")
        self.running_ms = RunningMeanStd(shape=(1,))

    def update(self, rewards):
        self.running_ms.update(np.asarray(rewards).reshape(-1, 1))

    @property
    def std(self):
        return float(np.sqrt(self.running_ms.var[0] + 1e-8))

    def normalize(self, rewards):
        return np.clip(np.asarray(rewards) / self.std, -self.clip, self.clip)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (np.sqrt(self.running_ms.var) + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


   