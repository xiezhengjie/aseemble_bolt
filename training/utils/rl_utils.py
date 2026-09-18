"""UR5e 插装定稿管线：共享基础 + 《插装任务残差 SAC-GAIL》数据结构与训练循环。

分区：
  1) 环境/基础组件：EpisodeStatsWrapper / wrap_frame_stack / RunningMeanStd ...
  2) 专家数据域：ExpertBuffer / ExpertDataManager / TensorBatchLoader
  3) 定稿三池（设计稿 §4）：ResidualReplayBuffer / GenWindowView / ExpertFlatSampler
  4) 采集（§7）：ResidualCollector（主循环/预填充同一实现）+ h 诊断
  5) 训练循环：train_sac_gail_residual（§7 数据流 + §8 更新频率）

旧 MountainCar 消融管线（train_sac_gail / ReplayBuffer / ...）已整体迁至
``utils/legacy_gail.py``（仍可用 RunningMeanStd）。

归一化口径（定稿 §5）：``diag(W)⁻¹(p−p_g)``、``diag(F_max)⁻¹F`` 在
``AssembleMuJoCoEnv`` 内完成；专家用 ``scripts/normalize_expert_data.py``
生成 ``*_norm.npz``。训练侧不再二次缩放。
"""
import os
import json
import time
import collections
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium import Wrapper
from gymnasium.wrappers import FrameStackObservation

from utils.checkpoint import has_weight



class EpisodeStatsWrapper(Wrapper):
    """回合结束时注入 info['final_info'] = {episodic_return, episodic_length, success, …}。

    适配 gym 标准环境（如 MountainCarContinuous-v0）与自定义环境（如 AssembleMuJoCoEnv）。
    success 优先取 env 的 info['success']；环境未提供该字段时不写入。
    插装任务额外转发 depth / position_error_xy / yaw_error，供 TensorBoard tasks/*。
    """

    _TASK_KEYS = ("depth", "position_error_xy", "yaw_error", "state")

    def __init__(self, env):
        super().__init__(env)
        self._ep_return = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        self._ep_return = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    @staticmethod
    def _as_scalar(v):
        arr = np.asarray(v)
        if arr.size == 0:
            return None
        x = arr.reshape(-1)[0]
        try:
            return float(x)
        except (TypeError, ValueError):
            return x

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
                    sv = self._as_scalar(info[k])
                    if sv is not None:
                        final_info[k] = sv
            info['final_info'] = final_info
        return obs, reward, terminated, truncated, info



def wrap_frame_stack(env, frame_stack, padding_type="zero"):
    """把单帧观测叠成 (T, *obs_shape)，给 GRU 用。

    对齐旧版环境 gru 模式：padding_type='zero' 时，reset 后不足的历史填 0，
    与 ExpertDataManager.convert_to_temporal_stack 一致。
    gymnasium 默认 padding_type='reset' 会重复首帧，不要用于本任务。
    """
    n = int(frame_stack)
    if n < 1:
        raise ValueError(f"frame_stack 必须 >= 1，当前: {frame_stack}")
    if n == 1:
        return env
    return FrameStackObservation(env, stack_size=n, padding_type=padding_type)



def _final_info_for_env(info, env_idx):
    """取出 env_idx 刚结束那一局的 final_info（VectorEnv autoreset 之后 info 已是新局）。"""
    if not isinstance(info, dict):
        return {}
    fi = info.get("final_info", info.get("_final_info"))
    if fi is None:
        return {}
    if isinstance(fi, dict):
        if "success" in fi or "episodic_length" in fi:
            out = {}
            for k, v in fi.items():
                try:
                    arr = np.asarray(v)
                    if arr.dtype == object and env_idx < len(arr) and isinstance(arr[env_idx], dict):
                        return dict(arr[env_idx])
                    if arr.shape == () or arr.ndim == 0:
                        out[k] = v
                    elif env_idx < arr.shape[0]:
                        out[k] = arr[env_idx]
                    else:
                        out[k] = v
                except (TypeError, ValueError, IndexError):
                    out[k] = v
            return out
        return fi
    try:
        if env_idx < len(fi) and isinstance(fi[env_idx], dict):
            item = fi[env_idx]
            inner = item.get("final_info") if "final_info" in item else None
            return dict(inner) if isinstance(inner, dict) else dict(item)
    except TypeError:
        pass
    return {}



def _episode_was_success(info, env_idx):
    """该环境本步结束的回合是否成功。缺字段视为否（仍写入 buffer_g）。"""
    item = _final_info_for_env(info, env_idx)
    if "success" in item:
        return bool(item["success"])
    succ = info.get("success") if isinstance(info, dict) else None
    if succ is None:
        return False
    arr = np.asarray(succ).reshape(-1)
    return bool(arr[env_idx]) if env_idx < len(arr) else False



def _task_metric_from_info(info, env_idx, key):
    """从 VectorEnv 的 final_info（优先）或顶层 info 取终止步任务量。"""
    item = _final_info_for_env(info, env_idx)
    if key in item:
        try:
            v = float(np.asarray(item[key]).reshape(-1)[0])
            return v if np.isfinite(v) else None
        except (TypeError, ValueError, IndexError):
            pass
    if not isinstance(info, dict) or key not in info:
        return None
    try:
        arr = np.asarray(info[key], dtype=np.float64).reshape(-1)
        if env_idx < len(arr) and np.isfinite(arr[env_idx]):
            return float(arr[env_idx])
    except (TypeError, ValueError):
        pass
    return None



def _finite_mean_list(values):
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(xs)) if xs else None



def _save_gail_ckpt(agent, disc, running_ms, save_model_dir, name, extra=None):
    if save_model_dir is None:
        return
    path = Path(save_model_dir) / name
    agent.save_model(path)
    disc.save_model(path)
    if running_ms is not None:
        running_ms.save_normalizer(path)
    if extra is not None:
        with open(path / "info.json", "w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2, ensure_ascii=False)



def _load_gail_ckpt(agent, disc, running_ms, save_model_dir, name="best_success_model"):
    if save_model_dir is None:
        return False
    path = Path(save_model_dir) / name
    if not has_weight(path, "policy_net"):
        return False
    agent.load_model(path)
    disc.load_model(path)
    if running_ms is not None:
        running_ms.load_normalizer(path)
    return True



def _handle_eval_checkpoint(
        stats, global_step, agent, disc, running_ms, save_model_dir, is_save_model,
        best_success):
    """评估后保存 final_model；成功率创新高时另存 best_success_model。"""
    success = float(stats['success_rate'])
    extra = {
        'success_rate': success,
        'global_step': int(global_step),
        'return_mean': float(stats.get('return_mean', 0.0)),
        'peak_force_mean': float(stats.get('peak_force_mean', 0.0)),
    }
    if is_save_model and save_model_dir:
        _save_gail_ckpt(agent, disc, running_ms, save_model_dir, "final_model", extra)
        if success > best_success:
            best_success = success
            _save_gail_ckpt(agent, disc, running_ms, save_model_dir, "best_success_model", extra)
    return best_success



def _episode_slices(n, dones):
    """把长度为 n 的序列按 done=1 切成 [start, end) 轨迹区间。"""
    n = int(n)
    if n <= 0:
        return []
    if dones is None:
        return [(0, n)]
    d = np.asarray(dones, dtype=np.float32).reshape(-1)
    if d.size < n:
        d = np.pad(d, (0, n - d.size))
    elif d.size > n:
        d = d[:n]
    ends = np.flatnonzero(d > 0.5)
    if ends.size == 0:
        return [(0, n)]
    slices = []
    start = 0
    for e in ends.tolist():
        e = int(e)
        if e >= start:
            slices.append((start, e + 1))
        start = e + 1
    if start < n:
        slices.append((start, n))
    return slices



def _concat_traj_sa(trajs):
    states = np.concatenate([t[0] for t in trajs], axis=0)
    actions = np.concatenate([t[1] for t in trajs], axis=0)
    return states, actions



def _sample_trajs_min_steps(n, get_traj, min_steps):
    """有放回抽完整轨迹，直到 (s,a) 总数 ≥ min_steps（至少 1 条）。"""
    if n == 0:
        raise ValueError("no trajectories to sample")
    min_steps = max(1, int(min_steps))
    picked = []
    total = 0
    guard = 0
    max_pick = max(n * 4, min_steps)
    while total < min_steps and guard < max_pick:
        s, a = get_traj(int(np.random.randint(0, n)))
        picked.append((s, a))
        total += int(s.shape[0])
        guard += 1
    return _concat_traj_sa(picked)


class RunningMeanStd:
    """使用 Welford 在线算法维护滑动均值和方差"""

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
        self._torch_stats.clear()
        # 展平到 (N, *shape)，统一处理
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.shape == self.mean.shape:
            x = x.reshape((1,) + self.mean.shape)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        
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
        """加载状态，支持 np.ndarray 或 list（从 JSON 反序列化）来源"""
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = int(state["count"])
        self._torch_stats = {}

    def save_normalizer(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        np.savez(_model_dir/"obs_normalizer.npz", **self.state_dict())

    def load_normalizer(self, model_dir):
        _model_path = Path(model_dir)/"obs_normalizer.npz"
        if _model_path.exists():
           with np.load(_model_path) as data:
               self.load_state_dict(data)


class ExpertBuffer:
    """numpy 预分配专家池，避免 random.sample(deque) 随容量线性变慢。"""

    def __init__(self, capacity, state_dim=None, action_dim=None):
        self.capacity = int(capacity)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._ready = False
        self._idx = 0
        self._size = 0
        self._ep_slices = []
        self._suffix_cache = {}

    def _build(self, state, action):
        s = np.asarray(state, dtype=np.float32)
        a = np.asarray(action, dtype=np.float32)
        if self._state_dim is not None:
            state_shape = tuple(self._state_dim) if isinstance(self._state_dim, (tuple, list)) else (int(self._state_dim),)
        else:
            state_shape = tuple(s.shape)
        ad = self._action_dim or int(a.shape[-1])
        self.states = np.zeros((self.capacity,) + state_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, ad), dtype=np.float32)
        self._ep_slices = []
        self._ready = True
        self._suffix_cache = {}

    def add(self, state, action):
        if not self._ready:
            self._build(state, action)
        i = self._idx
        self.states[i] = state
        self.actions[i] = action
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._ep_slices = [(0, self._size)] if self._size < self.capacity else [(0, self.capacity)]

    def load_arrays(self, states, actions, dones=None):
        """一次性写入全部专家样本，并按 dones 切成轨迹。"""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        n = int(states.shape[0])
        if n == 0:
            self.clear()
            return
        if n > self.capacity:
            self.capacity = n
        self._build(states[0], actions[0])
        self.states[:n] = states
        self.actions[:n] = actions
        self._size = n
        self._idx = n % self.capacity
        self._ep_slices = _episode_slices(n, dones)
        self._suffix_cache = {}

    def add_trajectory(self, states, actions):
        """追加一条完整轨迹并保留回合边界。GASILfD 把成功策略轨迹写入专家池用。

        只支持未绕环缓冲（``_idx == _size``）。专家池 capacity 通常远大于演示+自模仿总量。
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        n = int(states.shape[0])
        if n == 0:
            return
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"trajectory length mismatch: states={states.shape[0]} actions={actions.shape[0]}")
        if not self._ready:
            self._build(states[0], actions[0])
        if self._idx != self._size:
            raise RuntimeError(
                "ExpertBuffer 已绕环，无法追加轨迹。请增大 capacity。")
        if self._size + n > self.capacity:
            new_cap = max(self.capacity * 2, self._size + n)
            ns = np.zeros((new_cap,) + self.states.shape[1:], dtype=np.float32)
            na = np.zeros((new_cap, self.actions.shape[1]), dtype=np.float32)
            ns[:self._size] = self.states[:self._size]
            na[:self._size] = self.actions[:self._size]
            self.states, self.actions, self.capacity = ns, na, new_cap
        start = self._size
        self.states[start:start + n] = states
        self.actions[start:start + n] = actions
        self._ep_slices.append((start, start + n))
        self._size += n
        self._idx = self._size
        self._suffix_cache = {}

    def _suffix_indices(self, suffix_frac):
        """每条轨迹最后 suffix_frac 步的全局下标。"""
        key = round(float(suffix_frac), 6)
        if key <= 0.0:
            return np.arange(self._size, dtype=np.int64)
        cached = self._suffix_cache.get(key)
        if cached is not None:
            return cached
        parts = []
        for a, b in self._ep_slices:
            L = int(b - a)
            keep = max(1, int(np.ceil(L * key)))
            parts.append(np.arange(b - keep, b, dtype=np.int64))
        cached = np.concatenate(parts) if parts else np.arange(self._size, dtype=np.int64)
        self._suffix_cache[key] = cached
        return cached

    def sample(self, batch_size, suffix_frac=0.0, suffix_ratio=0.0):
        if self._size == 0:
            raise ValueError("Expert buffer is empty, cannot sample.")
        batch_size = int(batch_size)
        suffix_frac = float(suffix_frac)
        suffix_ratio = float(suffix_ratio)
        if suffix_frac <= 0.0 or suffix_ratio <= 0.0:
            indices = np.random.randint(0, self._size, size=batch_size)
        else:
            n_suf = int(round(batch_size * suffix_ratio))
            n_uni = batch_size - n_suf
            suf_pool = self._suffix_indices(suffix_frac)
            parts = []
            if n_uni > 0:
                parts.append(np.random.randint(0, self._size, size=n_uni))
            if n_suf > 0:
                parts.append(suf_pool[np.random.randint(0, len(suf_pool), size=n_suf)])
            indices = np.concatenate(parts)
            np.random.shuffle(indices)
        return self.states[indices], self.actions[indices]

    def n_trajectories(self):
        return len(self._ep_slices)

    def iter_trajectories(self):
        for a, b in self._ep_slices:
            yield self.states[a:b], self.actions[a:b]

    def sample_trajectories(self, min_steps=1, suffix_frac=0.0, suffix_ratio=0.0):
        """抽完整轨迹直到 (s,a) 总数 ≥ min_steps。

        ``suffix_ratio`` 比例的轨迹只保留末尾 ``suffix_frac``（插入段过采样）。
        """
        n = self.n_trajectories()
        if n == 0:
            raise ValueError("Expert buffer has no trajectories.")
        suffix_frac = float(suffix_frac)
        suffix_ratio = float(suffix_ratio)

        def _get(i):
            a, b = self._ep_slices[i]
            if suffix_frac > 0.0 and suffix_ratio > 0.0 and np.random.rand() < suffix_ratio:
                L = int(b - a)
                keep = max(1, int(np.ceil(L * suffix_frac)))
                a = b - keep
            return self.states[a:b], self.actions[a:b]

        return _sample_trajs_min_steps(n, _get, min_steps)

    def data(self):
        if not self._ready or self._size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if self._size < self.capacity:
            return self.states[:self._size], self.actions[:self._size]
        idx = self._idx
        return (np.concatenate([self.states[idx:], self.states[:idx]]),
                np.concatenate([self.actions[idx:], self.actions[:idx]]))

    def clear(self):
        self._idx = 0
        self._size = 0
        # 下次 add 按新维度重建（同一 manager 切换 raw/stack/rate 时）
        self._ready = False
        self.states = None
        self.actions = None
        self._ep_slices = []
        self._suffix_cache = {}

    def size(self):
        return self._size


class TensorBatchLoader:
    """内存张量按 batch 切片。每个 epoch 只做一次 shuffle + N/B 次切片，
    而不是 DataLoader 每个 batch 256 次 Python __getitem__。"""

    def __init__(self, states, actions, batch_size, shuffle=False, drop_last=False):
        self.states = states
        self.actions = actions
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __len__(self):
        n = self.states.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.states.shape[0]
        if self.shuffle:
            perm = torch.randperm(n, device=self.states.device)
            states = self.states[perm]
            actions = self.actions[perm]
        else:
            states = self.states
            actions = self.actions
        end = (n // self.batch_size) * self.batch_size if self.drop_last else n
        bs = self.batch_size
        for i in range(0, end, bs):
            yield states[i:i + bs], actions[i:i + bs]



class ExpertDataManager:
    """专家数据管理：加载 npz →（可选叠帧）→ ExpertBuffer。"""
    def __init__(self, capacity=100000):
        self.buffer = ExpertBuffer(capacity)

    def load_data(self, data_path, frame_stack=1, raw_obs_dim=10):
        """加载专家数据（npz 始终为单帧 raw）。

        :param data_path:    npz 文件路径
        :param frame_stack:  >1 时转成 (T, D) 时序矩阵，对齐 FrameStackObservation。
        :param raw_obs_dim:  原始单帧观测维度（默认 10）
        """
        if not os.path.exists(data_path):
            print("path is not exists")
            return
        if os.path.splitext(str(data_path))[1] != ".npz":
            print("file type error")
            return

        self.buffer.clear()
        data = np.load(data_path, allow_pickle=True)
        states = data["states"]
        actions = data["actions"]
        dones = data["dones"] if "dones" in data.files else None

        if states.ndim != 2 or states.shape[1] != raw_obs_dim:
            # 兼容旧 9 维专家数据
            if states.ndim == 2 and raw_obs_dim == 10 and states.shape[1] == 9:
                pass
            elif int(frame_stack) > 1:
                raise ValueError(
                    f"专家 states 期望单帧 dim={raw_obs_dim}，实际 shape={states.shape}。"
                    "请用原始录制 npz（未叠帧）再转换。"
                )

        if int(frame_stack) > 1:
            states = self.convert_to_temporal_stack(
                states, dones, frame_stack, states.shape[1]
            )

        self.buffer.load_arrays(states, actions, dones=dones)
        print(f"[ExpertData] loaded {len(states)} samples, "
              f"frame_stack={int(frame_stack)}, shape={states.shape[1:]}, "
              f"n_traj={self.buffer.n_trajectories()}")

    @staticmethod
    def convert_to_temporal_stack(states, dones=None, frame_stack=10, raw_obs_dim=10):
        """正时序帧矩阵 [obs_{t-T+1}, ..., obs_t]，shape (N, T, D)。

        episode 边界不足的历史零填充，与 FrameStackObservation(padding_type='zero') 一致。
        按 lag 做 numpy 切片，避免 O(N·T²) Python 循环。
        """
        del raw_obs_dim  # 以 states 实际宽度为准
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError(f"states 期望 (N, D)，实际 shape={states.shape}")
        n, dim = states.shape
        t = int(frame_stack)
        if t < 1:
            raise ValueError(f"frame_stack 必须 >= 1，当前: {frame_stack}")
        if n == 0:
            return np.zeros((0, t, dim), dtype=np.float32)

        if dones is None:
            dones = np.zeros(n, dtype=np.float32)
        else:
            dones = np.asarray(dones, dtype=np.float32).reshape(-1)
            if dones.shape[0] != n:
                raise ValueError(f"dones 长度 {dones.shape[0]} 与 states {n} 不一致")

        is_new = np.zeros(n, dtype=bool)
        is_new[0] = True
        if n > 1:
            is_new[1:] = dones[:-1] > 0.5
        ep_id = np.cumsum(is_new.astype(np.int32))

        out = np.zeros((n, t, dim), dtype=np.float32)
        idx = np.arange(n)
        for tau in range(t):
            lag = t - 1 - tau
            src = idx - lag
            ok = src >= 0
            ok[ok] = ep_id[src[ok]] == ep_id[idx[ok]]
            out[ok, tau] = states[src[ok]]
        return out

    def trans_dataloder(self, split, batch_size, device=None):
        """ split:[0~1], 训练split% 验证(1-split)%

        device 非空时把整份专家数据一次搬到该设备，按 batch 切片，
        避免 DataLoader 每个 batch 做 256 次 Python __getitem__。
        """
        states, actions = self.buffer.data()
        states_t = torch.from_numpy(np.ascontiguousarray(states)).float()
        actions_t = torch.from_numpy(np.ascontiguousarray(actions)).float()
        n = states_t.shape[0]
        train_size = int(split * n)
        perm = torch.randperm(n)
        train_idx, val_idx = perm[:train_size], perm[train_size:]
        if device is not None:
            dev = torch.device(device)
            states_t = states_t.to(dev)
            actions_t = actions_t.to(dev)
            train_idx = train_idx.to(dev)
            val_idx = val_idx.to(dev)
        train_loader = TensorBatchLoader(
            states_t[train_idx], actions_t[train_idx], batch_size, shuffle=True, drop_last=True)
        val_loader = TensorBatchLoader(
            states_t[val_idx], actions_t[val_idx], batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader


def orthogonal_init(module, gain=np.sqrt(2), bias=0.0):
    """
    正交初始化，支持三种传入方式：
      1. 单个 nn.Linear            -> 直接初始化
      2. nn.Sequential / 任意容器模块 -> 递归初始化内部所有 Linear
      3. 其他类型                  -> 显式报错，防止静默跳过
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        torch.nn.init.constant_(module.bias, bias)
    elif isinstance(module, torch.nn.Module):
        for child in module.children():
            orthogonal_init(child, gain=gain, bias=bias)
    else:
        raise TypeError(f"orthogonal_init 不支持的类型: {type(module)}")


def find_project_root():
    """查找项目根目录"""
    current = Path(__file__).resolve().parent
    
    # 向上查找，直到找到 setup.py 等项目特有文件
    for parent in [current] + list(current.parents):
        if (parent / 'setup.py').exists():
            return parent
        if (parent / 'requirements.txt').exists():
            return parent
    
    return current.parent.parent.parent



def rollout_eval(env, agent, n_episodes=20, running_ms=None,
                 deterministic=True, seed_offset=100000):
    """
    用给定策略在 env 上跑 n_episodes 条评估轨迹。

    :param env:            评估环境（独立实例，不与训练 env 复用）
    :param agent:          含 take_action 接口的智能体
    :param n_episodes:     评估轨迹条数
    :param obs_normalizer: RunningMeanStd 实例；仅 normalize，绝不 update
    :param deterministic:  True 关闭探索噪声（SAC 取 mean 动作）
    :param seed_offset:    评估种子起点，跨 checkpoint 固定同一批初始条件
    :return:               指标字典
    """
    # 评估环境（squeeze=True）走下方的逐回合原逻辑。

    def _done(terminated, truncated):
        t = np.asarray(terminated).reshape(-1)
        u = np.asarray(truncated).reshape(-1)
        flag = bool(t[0] or u[0])
        return flag

    returns, lengths, successes, peak_forces, final_states = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_return, ep_length, ep_peak_force = 0.0, 0, 0.0

        success = False
        while not done:
            action = _take_action(agent, obs, running_ms=running_ms, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = _done(terminated, truncated)
            ep_return += float(np.asarray(reward).reshape(-1)[0])
            ep_length += 1
            force = np.asarray(info.get("force", np.zeros(6)), dtype=np.float32).reshape(-1)
            ep_peak_force = max(ep_peak_force, float(np.linalg.norm(force[:6])))
            term_b = bool(np.asarray(terminated).reshape(-1)[0])
            if term_b:
                success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])

        returns.append(ep_return)
        lengths.append(ep_length)
        successes.append(1.0 if success else 0.0)
        peak_forces.append(ep_peak_force)
        final_states.append(int(np.asarray(info.get("state", -1)).reshape(-1)[0]))

    states, counts = np.unique(final_states, return_counts=True)
    return {
        'success_rate':    float(np.mean(successes)),
        'return_mean':     float(np.mean(returns)),
        'return_std':      float(np.std(returns)),
        'length_mean':     float(np.mean(lengths)),
        'peak_force_mean': float(np.mean(peak_forces)),
        'state_dist':      {int(s): int(c) for s, c in zip(states, counts)},
        'n_episodes':      int(n_episodes),
    }



def _take_action(agent, obs, running_ms=None, deterministic=True, base_only=False):
    """归一化后取**环境执行**动作；agent 未实现 deterministic 时降级。"""
    obs_in = running_ms.normalize(obs) if running_ms is not None else obs
    try:
        if base_only:
            executed = agent.take_action(obs_in, deterministic=deterministic, base_only=True)
        else:
            executed = agent.take_action(obs_in, deterministic=deterministic)
    except TypeError:
        executed = agent.take_action(obs_in)
    return np.asarray(executed, dtype=np.float32)


def _np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)



def _vec_reset(env, seed=None, options=None):
    """VectorEnv.reset。``seed`` 为 int 时 worker i 得到 seed+i；None 则沿用已有 np_random。"""
    kwargs = {}
    if options is not None:
        kwargs["options"] = options
    if seed is not None:
        kwargs["seed"] = int(seed)
    return env.reset(**kwargs)


class ResidualReplayBuffer:
    """定稿 §4.1：SAC 回放池，按 transition 存 (s, s', h, h', a^e, r_env, done, success)。

    - s / s'：物理尺度归一化后的叠帧观测 (T, D)（Lin：diag(W)⁻¹Δp / diag(F_max)⁻¹F）；
    - h / h'：rollout 时冻结 GRU 在归一化 s 上算出，顺手存；
    - a^e：clip 后的实际执行动作（不存 ã、不存判别器奖励，采样时现算）；
    - h 只在写入时计算，基座冻结 → 永不过期（铁律 1）。
    """

    def __init__(self, capacity, seq_len=8, raw_obs_dim=10, gru_hidden_dim=64,
                 action_dim=6):
        self.capacity = int(capacity)
        self._seq_len = int(seq_len)
        self._raw_obs_dim = int(raw_obs_dim)
        self._h_dim = int(gru_hidden_dim)
        self._action_dim = int(action_dim)
        obs_shape = (self._seq_len, self._raw_obs_dim)
        self.states = np.zeros((self.capacity,) + obs_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity,) + obs_shape, dtype=np.float32)
        self.h = np.zeros((self.capacity, self._h_dim), dtype=np.float32)
        self.h_next = np.zeros((self.capacity, self._h_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self._action_dim), dtype=np.float32)
        self.r_env = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        # ep_ends：回合边界（terminated|truncated），仅供回合切分/配对表；
        # dones 仅承载 TD 掩码（terminated）——超时按无限视野 bootstrap，
        # 否则 85% 超时回合会在孔区反复写入"价值=0"伪终点（Pardo et al. 2018）。
        self.ep_ends = np.zeros((self.capacity, 1), dtype=np.float32)
        self.successes = np.zeros((self.capacity, 1), dtype=np.float32)
        self._idx = 0
        self._size = 0

    def _ring_write(self, arr, data):
        """按当前写指针把 (N, ...) 批写入预分配 ``arr``，绕环时拆两截。"""
        n = int(data.shape[0])
        i, cap = self._idx, self.capacity
        if i + n <= cap:
            arr[i:i + n] = data
        else:
            n1 = cap - i
            arr[i:] = data[:n1]
            arr[:n - n1] = data[n1:]

    def add_batch(self, obs, next_obs, h, h_next, actions, r_env, dones, successes,
                  ep_ends=None):
        """一次写入 ``n_envs`` 条。obs/next_obs: (N, T, D)；h/h': (N, H)；a^e: (N, A)。
        dones = TD 掩码（terminated）；ep_ends = 回合边界（含 truncated），缺省同 dones。"""
        if ep_ends is None:
            ep_ends = dones
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        n = int(obs.shape[0])
        if n == 0:
            return
        h = np.asarray(h, dtype=np.float32).reshape(n, -1)
        h_next = np.asarray(h_next, dtype=np.float32).reshape(n, -1)
        actions = np.asarray(actions, dtype=np.float32)
        parts = (
            (self.states, obs),
            (self.next_states, next_obs),
            (self.h, h),
            (self.h_next, h_next),
            (self.actions, actions),
            (self.r_env, np.asarray(r_env, dtype=np.float32).reshape(n, 1)),
            (self.dones, np.asarray(dones, dtype=np.float32).reshape(n, 1)),
            (self.ep_ends, np.asarray(ep_ends, dtype=np.float32).reshape(n, 1)),
            (self.successes, np.asarray(successes, dtype=np.float32).reshape(n, 1)),
        )
        for arr, data in parts:
            self._ring_write(arr, data)
        self._idx = (self._idx + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size):
        if self._size == 0:
            raise ValueError("ResidualReplayBuffer 为空，无法采样")
        idx = np.random.randint(0, self._size, size=int(batch_size))
        return {
            "states": self.states[idx],
            "next_states": self.next_states[idx],
            "h": self.h[idx],
            "h_next": self.h_next[idx],
            "actions": self.actions[idx],
            "r_env": self.r_env[idx],
            "dones": self.dones[idx],
            "successes": self.successes[idx],
        }

    def size(self):
        return self._size



class GenWindowView:
    """定稿 §4.2：buffer_g = buffer_r 最近 N 条 transition 的滑动窗口视图。

    逻辑两池、物理一份数据：窗口随 buffer_r 写入指针推进（FIFO 覆盖旧样本），
    供 D 的生成样本（标签 1）1:1 采样；k = N // S。
    """

    def __init__(self, buffer_r, window_size):
        self.buffer_r = buffer_r
        self.window_size = int(window_size)

    def size(self):
        return min(self.window_size, self.buffer_r.size())

    def sample(self, batch_size):
        """从最近窗口均匀抽 (s_t, a^e_t) 当前帧供 D 使用。"""
        n = self.size()
        if n == 0:
            raise ValueError("GenWindowView 为空：先采集再更新 D")
        newest = self.buffer_r._idx  # 写入指针指向"下一条要写的位置" = 最新+1
        start = (newest - n) % self.buffer_r.capacity
        idx = (start + np.random.randint(0, n, size=int(batch_size))) % self.buffer_r.capacity
        s_cur = self.buffer_r.states[idx][:, -1, :]  # (T, D) 的当前帧
        a_exec = self.buffer_r.actions[idx]
        return s_cur, a_exec



class ExpertFlatSampler:
    """定稿 §4.3：专家池固定摊平，按 transition 均匀抽 (s^E, a^E)。

    专家数据加载时用与在线相同的归一化；不过 GRU/BC、不做残差换算（定稿 §2）。
    """

    def __init__(self, states, actions):
        self.states = np.asarray(states, dtype=np.float32)  # (N, T, D)
        self.actions = np.asarray(actions, dtype=np.float32)
        if self.states.ndim == 2:
            self.states = self.states[:, None, :]  # 单帧兼容
        self.n = int(self.states.shape[0])
        if self.n != self.actions.shape[0]:
            raise ValueError(
                f"专家 states/actions 数不一致: {self.n} vs {self.actions.shape[0]}")

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n, size=int(batch_size))
        return self.states[idx][:, -1, :], self.actions[idx]

    def size(self):
        return self.n



def _h_diagnostics(h_batch, name="h_norm"):
    """定稿 §5 诊断：h 各维 mean/std、|h|>0.95 饱和占比、有效秩。"""
    x = np.asarray(h_batch, dtype=np.float64)
    if x.size == 0:
        return {}
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    sat_frac = float((np.abs(x) > 0.95).mean())
    # 有效秩：奇异值熵 / log(d)
    s = np.linalg.svd(x - x.mean(axis=0, keepdims=True),
                      compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()) / x.shape[1])
    return {
        f"{name}_absmean": float(np.abs(mean).mean()),
        f"{name}_std_mean": float(std.mean()),
        f"{name}_sat_frac": sat_frac,
        f"{name}_eff_rank": eff_rank,
    }



class ResidualCollector:
    """定稿 §7 统一 rollout 收集器：残差混合策略与冻结 BC 走同一条写路径。

    每步写入 buffer_r：(s_t, s_{t+1}, h_t, h_{t+1}, a^e_t, r_env, done, success)
      - 环境已输出物理归一化观测，**buffer 直接存 s**（定稿 §4.1/§5）；
      - ``base_only=True``（预填充，§7.6）：只跑冻结 BC，ã=0；
      - 否则跑 a^e = clip(a_base + ã)（§7.1/§7.2），buffer 存 clip 后的 a^e；
      - h 由冻结 GRU 在 s 上现算；done 时 next_state 用 final_observation 校正；
      - success 只认成功终止；超时/失败终止不算（§4.1）。
    """

    def __init__(self, agent, buffer_r):
        self.agent = agent
        self.buffer_r = buffer_r

    def run(self, env, n_steps, base_only=False, desc="collect",
            track_episodes=False):
        """采 ``n_steps`` 步；``track_episodes=True`` 时返回本批回合统计列表。"""
        agent, buffer_r = self.agent, self.buffer_r
        n_envs = int(getattr(env, "num_envs", 1))
        obs, _ = _vec_reset(env, seed=None)  # 沿用已有 np_random，不重播
        obs = _np(obs).astype(np.float32)
        ep_ret = np.zeros(n_envs, dtype=np.float64)
        ep_len = np.zeros(n_envs, dtype=np.int64)
        episodes = []
        with tqdm(total=int(n_steps), desc=desc, dynamic_ncols=True, mininterval=0.5) as pbar:
            while pbar.n < pbar.total:
                if base_only:
                    a_exec, h_now = agent.take_action_base_only(obs)
                else:
                    a_exec, _, h_now = agent.act_collect(obs)
                next_obs, env_r, terminated, truncated, info = env.step(a_exec)
                next_obs = _np(next_obs).astype(np.float32)
                env_r = _np(env_r).reshape(n_envs).astype(np.float32)
                terminated = _np(terminated).reshape(n_envs).astype(bool)
                truncated = _np(truncated).reshape(n_envs).astype(bool)
                dones = terminated | truncated

                # next_state：done 行用 final_observation 校正（已是归一化观测）
                next_states = next_obs.copy()
                final_obs_arr = info.get("final_observation") if np.any(dones) else None
                if final_obs_arr is not None:
                    for j in np.flatnonzero(dones):
                        fo_j = final_obs_arr[j]
                        if fo_j is not None:
                            next_states[j] = np.asarray(fo_j, dtype=np.float32)
                h_next = agent.base_hidden_batch(next_states)

                # success：autoreset 后 info["success"] 已是新局值，必须走
                # _episode_was_success；失败终止（状态 4）也是 terminated。
                success_flags = np.zeros(n_envs, dtype=np.float32)
                for j in np.flatnonzero(dones):
                    success_flags[j] = float(
                        terminated[j] and _episode_was_success(info, int(j)))

                buffer_r.add_batch(obs, next_states, h_now, h_next, a_exec,
                                   env_r, terminated.astype(np.float32), success_flags,
                                   ep_ends=dones.astype(np.float32))

                if track_episodes:
                    ep_ret += env_r
                    ep_len += 1
                    if np.any(dones):
                        for j in np.flatnonzero(dones):
                            jj = int(j)
                            episodes.append({
                                "episodic_return": float(ep_ret[j]),
                                "episodic_length": int(ep_len[j]),
                                "success": bool(success_flags[j] > 0.5),
                                "depth": _task_metric_from_info(info, jj, "depth"),
                                "position_error_xy": _task_metric_from_info(
                                    info, jj, "position_error_xy"),
                                "yaw_error": _task_metric_from_info(
                                    info, jj, "yaw_error"),
                            })
                        ep_ret[dones] = 0.0
                        ep_len[dones] = 0
                obs = next_obs
                pbar.update(n_envs)
        return episodes if track_episodes else None



def train_sac_gail_residual(env, eval_env, agent, disc, buffer_r, buffer_g, buffer_e,
                            total_timesteps, seed, sac_updates=40, disc_updates=10,
                            batch_size=1024, disc_batch_size=1024,
                            success_reward=100.0,
                            eval_interval=56_000, eval_episodes=30,
                            save_model_dir=None, log_dir=None, run_name=None,
                            is_save_model=True, is_draw=True,
                            prefill_steps=0, max_steps_lr=2_000_000,
                            steps_per_iter=None):
    """定稿版残差 SAC-GAIL 训练循环（§7 数据流 + §8 更新频率）。

    轮内顺序：采集 S=5600 步 → SAC 40 次×1024（用上一轮的 D）→ D 10 次×1024，
    一轮滞后是 GAN 式交替的标准做法（§7.7）。

    - 奖励 r̃ = softplus(−z)/20 ∈[0,1]，用**当前** D 逐批现算（buffer 存的奖励不作数）；
    - y = r̃ + R_succ·1{success} + γ(1−done)[Q_targ − α log f(ã′)]，
      成功终止不 bootstrap（§6）；默认 R_succ≈γ/(1−γ)≈100；
    - D 反标签更新：生成样本 ∼ buffer_g（标签 1），专家样本 ∼ buffer_e（标签 0），
      1:1，GP 10.0 + 熵正则 0.001（§0/§7.5）；
    - buffer_g 是 buffer_r 最近 N=k·S 条的 FIFO 窗口视图（§4.2）。
    - 观测已在环境 / ``*_norm.npz`` 中归一化，本循环不再二次缩放（§5）。

    ``agent`` / ``disc``：``ResidualSAC`` / ``MarkovDiscriminator``。
    ``steps_per_iter``：每轮采集步数 S；None 时按 14 环境×400 步推。
    """
    if getattr(env, "num_envs", None) is None:
        raise TypeError("train_sac_gail_residual 需要 VectorEnv（num_envs>=1）")
    n_envs = int(getattr(env, "num_envs", 1))
    if steps_per_iter is None:
        steps_per_iter = max(n_envs, int(round(
            total_timesteps / max(1, round(total_timesteps / (n_envs * 400))))))
    steps_per_iter = max(n_envs, int(steps_per_iter))
    n_iters = max(1, int(round(total_timesteps / steps_per_iter)))
    steps_per_iter = max(n_envs, int(round(total_timesteps / n_iters)))

    if not run_name:
        run_name = f"sac_gail_residual__{seed}__{int(time.time())}"
    writer = SummaryWriter(Path(log_dir) / run_name)
    return_list = []
    success_hist = collections.deque(maxlen=50)
    best_success = 0.0
    global_step = 0

    print(f"[train] total={total_timesteps} iters={n_iters} "
          f"steps/iter={steps_per_iter} (n_envs={n_envs}), "
          f"SAC {sac_updates}x{batch_size} / D {disc_updates}x{disc_batch_size}, "
          f"R_succ={success_reward:g}, buffer_g window={buffer_g.window_size}")

    # ---------- 预填充（§7.6）：冻结 BC rollout ----------
    if prefill_steps > 0:
        writer.add_scalar("charts/prefill_steps", float(prefill_steps), 0)
        ResidualCollector(agent, buffer_r).run(
            env, prefill_steps, base_only=True, desc="prefill(BC)")

    # ---------- 主循环：采集 → SAC → D ----------
    collector = ResidualCollector(agent, buffer_r)
    recent_avg_window = 10  # 对齐 legacy：avg_episodic_return 近窗均值
    for it in range(1, n_iters + 1):
        t_iter = time.time()
        episodes = collector.run(env, steps_per_iter, desc=f"Iter:{it:d}",
                                 track_episodes=True)
        collect_dt = max(time.time() - t_iter, 1e-6)
        global_step += steps_per_iter
        for ep in episodes:
            return_list.append(ep["episodic_return"])
            success_hist.append(1.0 if ep["success"] else 0.0)

        # ---------- SAC：40 次 × batch 1024，奖励由当前 D 现算 ----------
        avg_info = {}
        for ui in range(sac_updates):
            batch = buffer_r.sample(batch_size)
            s_cur = batch["states"][:, -1, :]
            a_exec = batch["actions"]
            r_tilde = disc.predict_rewards(s_cur, a_exec, to_numpy=True).reshape(-1, 1)
            rewards = r_tilde + success_reward * batch["successes"]
            transition_dict = dict(batch)
            transition_dict["rewards"] = rewards
            info = agent.update(transition_dict, log_info=(ui == sac_updates - 1))
            if info:
                avg_info.update(info)
        agent.lr_decay(global_step, max_steps=max_steps_lr)

        # ---------- D：10 次 × batch 1024，生成/专家 1:1（反标签） ----------
        for di in range(disc_updates):
            g_s, g_a = buffer_g.sample(disc_batch_size)
            e_s, e_a = buffer_e.sample(disc_batch_size)
            d_info = disc.update(e_s, e_a, g_s, g_a, log_info=(di == disc_updates - 1))
            if d_info:
                avg_info.update(d_info)
        disc.lr_decay(global_step, max_steps=max_steps_lr)

        # ---------- 日志 ----------
        sps = int(steps_per_iter / collect_dt)
        if is_draw:
            m = _h_diagnostics(buffer_r.h[:: max(1, buffer_r.size() // 512)])
            for k, v in m.items():
                writer.add_scalar(f"diag/{k}", v, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/buffer_r", float(buffer_r.size()), global_step)
            writer.add_scalar("charts/buffer_g_window", float(buffer_g.size()), global_step)
            for k, v in avg_info.items():
                writer.add_scalar(f"losses/{k}", v, global_step)
        if episodes:
            mean_ret = float(np.mean([e["episodic_return"] for e in episodes]))
            mean_len = float(np.mean([e["episodic_length"] for e in episodes]))
            succ_rate = float(np.mean(success_hist)) if success_hist else 0.0
            # 残差管线采集只累加 env 奖励；GAIL 奖励在 update 时现算，无在线回合和
            avg_ret = float(np.mean(return_list[-recent_avg_window:]))
            print(f"[iter {it}/{n_iters} @ {global_step}] "
                  f"ret={mean_ret:.1f} avg{recent_avg_window}={avg_ret:.1f} "
                  f"len={mean_len:.1f} succ50={succ_rate:.1%} "
                  f"D_acc={avg_info.get('disc_acc', float('nan')):.3f} SPS={sps}")
            if is_draw:
                writer.add_scalar("charts/episodic_return", mean_ret, global_step)
                writer.add_scalar("charts/avg_episodic_return", avg_ret, global_step)
                writer.add_scalar("charts/episodic_env_return", mean_ret, global_step)
                writer.add_scalar("charts/episodic_length", mean_len, global_step)
                writer.add_scalar("charts/success_rate_50ep", succ_rate, global_step)
                depth_v = _finite_mean_list(e.get("depth") for e in episodes)
                xy_v = _finite_mean_list(e.get("position_error_xy") for e in episodes)
                yaw_v = _finite_mean_list(e.get("yaw_error") for e in episodes)
                if depth_v is not None:
                    writer.add_scalar("tasks/depth (m)", depth_v, global_step)
                if xy_v is not None:
                    writer.add_scalar("tasks/position_error_xy (m)", xy_v, global_step)
                if yaw_v is not None:
                    writer.add_scalar("tasks/yaw_error (deg)", yaw_v, global_step)

        # ---------- 周期评估 ----------
        if eval_env is not None and global_step % eval_interval < steps_per_iter:
            stats = rollout_eval(eval_env, agent, n_episodes=eval_episodes)
            best_success = _handle_eval_checkpoint(
                stats, global_step, agent, disc, None, save_model_dir,
                is_save_model, best_success)
            if is_draw:
                writer.add_scalar("eval/success_rate", stats["success_rate"], global_step)
                writer.add_scalar("eval/return_mean", stats["return_mean"], global_step)
                writer.add_scalar("eval/peak_force_mean", stats["peak_force_mean"], global_step)
            print(f"[eval@{global_step}] success={stats['success_rate']:.1%} "
                  f"ret={stats['return_mean']:.1f}±{stats['return_std']:.1f} "
                  f"peakF={stats['peak_force_mean']:.1f}N states={stats['state_dist']}")

    # ---------- 结束：保存与终检 ----------
    if is_save_model and save_model_dir:
        agent.save_model(Path(save_model_dir) / "final_model")
        disc.save_model(Path(save_model_dir) / "final_model")
        final_info = {
            "best_success": float(best_success),
            "return": float(return_list[-1]) if return_list else None,
            "global_step": int(global_step),
            "n_episodes": int(len(return_list)),
        }
        (Path(save_model_dir) / "final_model").mkdir(parents=True, exist_ok=True)
        with open(Path(save_model_dir) / "final_model" / "info.json", "w",
                  encoding="utf-8") as f:
            json.dump(final_info, f, indent=2, ensure_ascii=False)

    # 恢复成功率最高的 checkpoint 再终检（对齐 train_sac_gail 的语义）
    restored_best = False
    if is_save_model and save_model_dir:
        best_dir = Path(save_model_dir) / "best_success_model"
        if has_weight(best_dir, "policy_net"):
            agent.load_model(best_dir)
            disc.load_model(best_dir)
            restored_best = True

    if eval_env is not None:
        final_stats = rollout_eval(eval_env, agent, n_episodes=max(30, eval_episodes),
                                   seed_offset=200000)
        ckpt_tag = "best_success_model" if restored_best else "last_weights"
        print(f"[final eval @ {ckpt_tag}] success={final_stats['success_rate']:.1%} "
              f"ret={final_stats['return_mean']:.1f}±{final_stats['return_std']:.1f} "
              f"peakF={final_stats['peak_force_mean']:.1f}N")
        eval_env.close()

    env.close()
    writer.close()
    return np.asarray(return_list, dtype=np.float32)


def _ss_pair_table(buffer_r, k):
    """预计算 K 步配对表（每轮一次）：pair_next[i] = states[i+k] 当前帧
    （区间无 done 时），否则退回 next_states[i]；kvalid 标记 K 步有效性。"""
    size = buffer_r.size()
    if size == 0:
        return np.zeros((0, buffer_r.states.shape[-1]), dtype=np.float32), \
            np.zeros(0, dtype=bool)
    k = int(k)
    states = buffer_r.states[:size, -1, :]
    next_states = buffer_r.next_states[:size, -1, :]
    dones = buffer_r.ep_ends[:size].reshape(-1)
    pair_next = next_states.copy()
    kvalid = np.zeros(size, dtype=bool)
    if k > 1 and size > k:
        cs = np.concatenate([[0], np.cumsum(dones)])
        t = np.arange(0, size - k)
        ok = (cs[t + k] - cs[t]) == 0
        pair_next[t[ok]] = states[t[ok] + k]
        kvalid[t[ok]] = True
    else:
        kvalid[:-1] = dones[:-1] == 0
    return pair_next, kvalid


def _ss_window_pairs(buffer_r, window_size, batch_size, pair_next, kvalid):
    """从 buffer_r 最近 window_size 条里均匀抽 (s_t, s_{t+K}) 当前帧对
    （仅供 D 负类兜底；正常路径用结局条件采样）。"""
    size = buffer_r.size()
    n = min(int(window_size), size)
    if n == 0:
        raise ValueError("buffer 为空：先采集再更新 D")
    lo = size - n
    local = np.random.randint(0, n, size=int(batch_size))
    idx = lo + local
    return buffer_r.states[idx, -1, :], pair_next[idx]


def _success_tail_pool(buffer_r, tail=32):
    """成功回合尾部索引池：每条成功转移及其同回合前 tail 步（供上采样）。"""
    size = buffer_r.size()
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    succ = buffer_r.successes[:size].reshape(-1) > 0.5
    dones = buffer_r.ep_ends[:size].reshape(-1) > 0.5
    pool = []
    for idx in np.where(succ)[0]:
        pool.append(idx)
        j = idx - 1
        for _ in range(int(tail)):
            if j < 0 or dones[j]:
                break
            pool.append(j)
            j -= 1
    return np.asarray(pool, dtype=np.int64)


def _gather_batch(buffer_r, idx):
    return {
        "states": buffer_r.states[idx],
        "next_states": buffer_r.next_states[idx],
        "h": buffer_r.h[idx],
        "h_next": buffer_r.h_next[idx],
        "actions": buffer_r.actions[idx],
        "r_env": buffer_r.r_env[idx],
        "dones": buffer_r.dones[idx],
        "successes": buffer_r.successes[idx],
    }


def _episode_outcome_table(buffer_r):
    """outcome[i]：transition i 所在回合的结局（1=成功，0=失败/超时，-1=未终止）。

    向量化：对每个 i 找其后第一个 done，该 done 的 success 即本回合结局。
    buffer 尾部尚未终止的回合标 -1（D 训练时排除）。
    """
    size = buffer_r.size()
    if size == 0:
        return np.zeros(0, dtype=np.float32)
    dones = buffer_r.ep_ends[:size].reshape(-1) > 0.5
    succ = buffer_r.successes[:size].reshape(-1) > 0.5
    done_idx = np.where(dones)[0]
    outcome = np.full(size, -1.0, dtype=np.float32)
    if len(done_idx) == 0:
        return outcome
    pos = np.searchsorted(done_idx, np.arange(size), side="left")
    valid = pos < len(done_idx)
    outcome[valid] = succ[done_idx[pos[valid]]].astype(np.float32)
    return outcome


def train_sac_gail_residual_ss(env, eval_env, agent, disc, buffer_r, expert_ss,
                               total_timesteps, seed, sac_updates=20, disc_updates=5,
                               batch_size=512, disc_batch_size=512,
                               success_reward=332.0, disc_reward_coef=1.0,
                               buffer_g_window=None,
                               pair_k=8,
                               success_tail=32, succ_frac=0.125,
                               eval_interval=12_000, eval_episodes=10,
                               save_model_dir=None, log_dir=None, run_name=None,
                               is_save_model=True, is_draw=True,
                               prefill_steps=0, max_steps_lr=2_000_000,
                               steps_per_iter=None):
    """(s,s′) 版残差 SAC-GAIL 训练循环（v6-v8）。

    演进链（均有实测依据）：
    - v2 奖励 r̃ = D(s_t, s_{t+1}) 现算（消除 (s,a) 型奖励谷地）；
    - v3 K 步对 (s_t, s_{t+K})：单步不可分（BC 失败是时序停滞），K=8 可分；
    - v4 奖励不 ÷20（有效对比度 0.01→~1/步，熵项不再主导）；
    - v5 结局条件 D：正类 = 专家 + 在线成功回合对，负类 = 失败回合对
      （纯二分类置信度被"区域可分性"主导 → 接近带 r̃>>孔区 → 策略务农；
      结局条件后务农环整体落入负类，自我纠正）；
    - v6 奖励由 D 的 EMA 副本产出（阻尼 D-策略追逐的极限环）；
    - v7 SQIL 化：r̃ = C·1{z_EMA<0}，移除可务农的奖励幅度；
    - v8 可切纯稀疏：disc_reward_coef=0 时 r = R_succ·1{success}，
      奖励完全平稳，从结构上消除判别器非平稳性与 D-policy 极限环。
    """
    if getattr(env, "num_envs", None) is None:
        raise TypeError("train_sac_gail_residual_ss 需要 VectorEnv（num_envs>=1）")
    n_envs = int(getattr(env, "num_envs", 1))
    if steps_per_iter is None:
        steps_per_iter = n_envs * 400
    steps_per_iter = max(n_envs, int(steps_per_iter))
    n_iters = max(1, int(round(total_timesteps / steps_per_iter)))
    if buffer_g_window is None:
        buffer_g_window = 3 * steps_per_iter

    e_ss_s, e_ss_n = expert_ss
    if not run_name:
        run_name = f"sac_gail_residual_ss__{seed}__{int(time.time())}"
    writer = SummaryWriter(Path(log_dir) / run_name)
    return_list = []
    success_hist = collections.deque(maxlen=50)
    best_success = 0.0
    global_step = 0

    print(f"[train-ss] total={total_timesteps} iters={n_iters} "
          f"steps/iter={steps_per_iter} (n_envs={n_envs}), "
          f"SAC {sac_updates}x{batch_size} / D {disc_updates}x{disc_batch_size}, "
          f"R_succ={success_reward:g} disc_coef={disc_reward_coef:g} "
          f"window={buffer_g_window} "
          f"K={pair_k} succ_tail={success_tail} succ_frac={succ_frac}", flush=True)

    if prefill_steps > 0:
        writer.add_scalar("charts/prefill_steps", float(prefill_steps), 0)
        ResidualCollector(agent, buffer_r).run(
            env, prefill_steps, base_only=True, desc="prefill(BC)")

    collector = ResidualCollector(agent, buffer_r)
    recent_avg_window = 10
    for it in range(1, n_iters + 1):
        t_iter = time.time()
        episodes = collector.run(env, steps_per_iter, desc=f"Iter:{it:d}",
                                 track_episodes=True)
        collect_dt = max(time.time() - t_iter, 1e-6)
        global_step += steps_per_iter
        for ep in episodes:
            return_list.append(ep["episodic_return"])
            success_hist.append(1.0 if ep["success"] else 0.0)

        # 成功回合尾部上采样池 + K 步配对表 + 回合结局表（每轮重建）
        pool = _success_tail_pool(buffer_r, tail=success_tail)
        n_succ = min(int(batch_size * succ_frac), len(pool))
        pair_next, kvalid = _ss_pair_table(buffer_r, pair_k)
        outcome = _episode_outcome_table(buffer_r)

        # ---------- SAC：奖励由当前 D 现算（(s, s_{t+K}) 版，EMA 网络） ----------
        avg_info = {}
        for ui in range(sac_updates):
            n_uni = batch_size - n_succ
            idx = np.random.randint(0, buffer_r.size(), size=n_uni)
            if n_succ > 0:
                idx = np.concatenate([idx, np.random.choice(pool, size=n_succ, replace=True)])
            batch = _gather_batch(buffer_r, idx)
            s_cur = batch["states"][:, -1, :]
            r_tilde = disc.predict_rewards(s_cur, pair_next[idx], to_numpy=True).reshape(-1, 1)
            batch["rewards"] = (float(disc_reward_coef) * r_tilde
                                + success_reward * batch["successes"])
            info = agent.update(batch, log_info=(ui == sac_updates - 1))
            if info:
                avg_info.update(info)
        agent.lr_decay(global_step, max_steps=max_steps_lr)

        # ---------- D（v5 结局条件）：正类 = 专家对 + 在线成功回合对；
        # 负类 = 最近窗口失败回合对 ----------
        size_now = buffer_r.size()
        win_lo = max(0, size_now - buffer_g_window)
        win_idx = np.arange(win_lo, size_now)
        win_valid = kvalid[win_lo:size_now]
        fail_pool = win_idx[win_valid & (outcome[win_lo:size_now] == 0)]
        succ_pair_pool = np.where(kvalid & (outcome == 1))[0]
        n_pos_each = disc_batch_size // 4          # 专家、在线成功各 1/4
        n_neg = disc_batch_size - 2 * n_pos_each   # 失败 1/2
        for di in range(disc_updates):
            ei = np.random.randint(0, len(e_ss_s), size=n_pos_each)
            if len(succ_pair_pool) > 0:
                si = np.random.choice(succ_pair_pool, size=n_pos_each, replace=True)
                p_s = np.concatenate([e_ss_s[ei], buffer_r.states[si, -1, :]])
                p_n = np.concatenate([e_ss_n[ei], pair_next[si]])
            else:
                ei2 = np.random.randint(0, len(e_ss_s), size=n_pos_each)
                p_s = np.concatenate([e_ss_s[ei], e_ss_s[ei2]])
                p_n = np.concatenate([e_ss_n[ei], e_ss_n[ei2]])
            if len(fail_pool) > 0:
                fi = np.random.choice(fail_pool, size=n_neg, replace=True)
                g_s = buffer_r.states[fi, -1, :]
                g_sn = pair_next[fi]
            else:  # 极早期无失败样本：退回窗口随机负类
                g_s, g_sn = _ss_window_pairs(buffer_r, buffer_g_window, n_neg,
                                             pair_next=pair_next, kvalid=kvalid)
            d_info = disc.update(p_s, p_n, g_s, g_sn,
                                 log_info=(di == disc_updates - 1))
            if d_info:
                avg_info.update(d_info)
        disc.lr_decay(global_step, max_steps=max_steps_lr)

        # ---------- 日志 ----------
        sps = int(steps_per_iter / collect_dt)
        if is_draw:
            m = _h_diagnostics(buffer_r.h[:: max(1, buffer_r.size() // 512)])
            for k, v in m.items():
                writer.add_scalar(f"diag/{k}", v, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/buffer_r", float(buffer_r.size()), global_step)
            writer.add_scalar("charts/succ_pool", float(len(pool)), global_step)
            writer.add_scalar("charts/d_pos_pool", float(len(succ_pair_pool)), global_step)
            writer.add_scalar("charts/d_neg_pool", float(len(fail_pool)), global_step)
            for k, v in avg_info.items():
                writer.add_scalar(f"losses/{k}", v, global_step)
        if episodes:
            mean_ret = float(np.mean([e["episodic_return"] for e in episodes]))
            mean_len = float(np.mean([e["episodic_length"] for e in episodes]))
            succ_rate = float(np.mean(success_hist)) if success_hist else 0.0
            avg_ret = float(np.mean(return_list[-recent_avg_window:]))
            print(f"[iter {it}/{n_iters} @ {global_step}] "
                  f"ret={mean_ret:.1f} avg{recent_avg_window}={avg_ret:.1f} "
                  f"len={mean_len:.1f} succ50={succ_rate:.1%} "
                  f"pool={len(pool)} D_acc={avg_info.get('disc_acc', float('nan')):.3f} "
                  f"SPS={sps}", flush=True)
            if is_draw:
                writer.add_scalar("charts/episodic_return", mean_ret, global_step)
                writer.add_scalar("charts/avg_episodic_return", avg_ret, global_step)
                writer.add_scalar("charts/episodic_env_return", mean_ret, global_step)
                writer.add_scalar("charts/episodic_length", mean_len, global_step)
                writer.add_scalar("charts/success_rate_50ep", succ_rate, global_step)
                depth_v = _finite_mean_list(e.get("depth") for e in episodes)
                xy_v = _finite_mean_list(e.get("position_error_xy") for e in episodes)
                yaw_v = _finite_mean_list(e.get("yaw_error") for e in episodes)
                if depth_v is not None:
                    writer.add_scalar("tasks/depth (m)", depth_v, global_step)
                if xy_v is not None:
                    writer.add_scalar("tasks/position_error_xy (m)", xy_v, global_step)
                if yaw_v is not None:
                    writer.add_scalar("tasks/yaw_error (deg)", yaw_v, global_step)

        # ---------- 周期评估 ----------
        if eval_env is not None and global_step % eval_interval < steps_per_iter:
            stats = rollout_eval(eval_env, agent, n_episodes=eval_episodes)
            best_success = _handle_eval_checkpoint(
                stats, global_step, agent, disc, None, save_model_dir,
                is_save_model, best_success)
            if is_draw:
                writer.add_scalar("eval/success_rate", stats["success_rate"], global_step)
                writer.add_scalar("eval/return_mean", stats["return_mean"], global_step)
                writer.add_scalar("eval/peak_force_mean", stats["peak_force_mean"], global_step)
            print(f"[eval@{global_step}] success={stats['success_rate']:.1%} "
                  f"ret={stats['return_mean']:.1f}±{stats['return_std']:.1f} "
                  f"peakF={stats['peak_force_mean']:.1f}N", flush=True)

    if is_save_model and save_model_dir:
        agent.save_model(Path(save_model_dir) / "final_model")
        disc.save_model(Path(save_model_dir) / "final_model")
    env.close()
    writer.close()
    return np.asarray(return_list, dtype=np.float32)
