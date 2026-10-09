from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BaseBuffer:
    """numpy 环形回放池。"""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._idx = 0
        self._size = 0
        self._arrays: Optional[dict] = None

    def add_batch(self, **extra):
        values = {key: np.asarray(value, dtype=np.float32) for key, value in extra.items()}
        n = next(iter(values.values())).shape[0]
        if self._arrays is not None:
            if set(values) != set(self._arrays):
                raise ValueError("回放池字段在首次写入后不能改变")
            if any(value.shape[1:] != self._arrays[key].shape[1:] for key, value in values.items()):
                raise ValueError("回放池字段 shape 在首次写入后不能改变")
        if n == 0:
            return
        if self._arrays is None:
            self._arrays = {
                key: np.zeros((self.capacity,) + value.shape[1:], dtype=np.float32)
                for key, value in values.items()
            }
            for key, array in self._arrays.items():
                setattr(self, key, array)

        start = max(0, n - self.capacity)
        indices = (self._idx + np.arange(start, n)) % self.capacity
        for key, value in values.items():
            self._arrays[key][indices] = value[start:]
        self._idx = int((self._idx + n) % self.capacity)
        self._size = min(self.capacity, self._size + n)

    def add(self, **extra):
        self.add_batch(**{key: np.asarray(value)[None] for key, value in extra.items()})

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=int(batch_size))
        return {key: value[indices] for key, value in self._arrays.items()}

    def size(self):
        return self._size

    def clear(self):
        if self._arrays is not None:
            for key in self._arrays:
                delattr(self, key)
        self._size = 0
        self._idx = 0
        self._arrays = None

    @property
    def buffer(self):
        return self._arrays


class ExpertBuffer(BaseBuffer):
    """专家池，字段固定为 obs / actions / dones。"""

    def sequence_data(self):
        """按写入顺序返回有效数据。"""
        indices = (self._idx - self._size + np.arange(self._size)) % self.capacity
        return {key: value[indices] for key, value in self.buffer.items()}

    def add_batch(self, obs, actions, dones):
        return super().add_batch(obs=obs, actions=actions, dones=dones)

    def add(self, obs, action, done):
        return self.add_batch(
            obs=np.asarray(obs)[None],
            actions=np.asarray(action)[None],
            dones=np.asarray([done]),
        )


class ReplayBuffer(BaseBuffer):
    """标准 RL 回放池。"""

    def add_batch(self, obs, actions, next_obs, rewards, dones, successes):
        return super().add_batch(
            obs=obs, actions=actions, rewards=rewards,
            next_obs=next_obs, dones=dones, successes=successes,
        )

    def add(self, obs, action, reward, next_obs, done, success):
        return self.add_batch(
            obs=np.asarray(obs)[None],
            actions=np.asarray(action)[None],
            rewards=np.asarray([reward]),
            next_obs=np.asarray(next_obs)[None],
            dones=np.asarray([done]),
            successes=np.asarray([success]),
        )


class ResidualReplayBuffer(BaseBuffer):
    """残差网络回放池；当前和下一观测分别拼接对应的基座动作。"""

    def add_batch(self, obs, base_actions, actions, res_actions,
                  rewards, next_obs, next_base_actions, dones, successes):
        self.raw_obs_dim = np.asarray(obs).shape[-1]
        return super().add_batch(
            obs=np.concatenate([obs, base_actions], axis=-1),
            actions=actions,    # 判别器输入
            res_actions=res_actions,
            rewards=rewards,
            next_obs=np.concatenate([next_obs, next_base_actions], axis=-1),
            dones=dones,
            successes=successes,
        )

    def add(self, obs, base_action, action, res_action, reward, next_obs,
            next_base_action, done, success):
        return self.add_batch(
            obs=np.asarray(obs)[None],
            base_actions=np.asarray(base_action)[None],
            actions=np.asarray(action)[None],
            res_actions=np.asarray(res_action)[None],
            rewards=np.asarray([reward]),
            next_obs=np.asarray(next_obs)[None],
            next_base_actions=np.asarray(next_base_action)[None],
            dones=np.asarray([done]),
            successes=np.asarray([success]),
        )


class GenWindowView:
    """buffer 的滑动窗口视图：取最近 N 条 transition。"""

    def __init__(self, buffer_r, window_size, current_frame=True):
        self.buffer_r = buffer_r
        self.window_size = int(window_size)
        self.current_frame = bool(current_frame)

    def size(self):
        return min(self.window_size, self.buffer_r.size())

    def sample(self, batch_size):
        n = self.size()
        newest = self.buffer_r._idx
        start = (newest - n) % self.buffer_r.capacity
        idx = (start + np.random.randint(0, n, size=int(batch_size))) % self.buffer_r.capacity

        obs = self.buffer_r.obs[idx]
        if isinstance(self.buffer_r, ResidualReplayBuffer):
            obs = obs[..., :self.buffer_r.raw_obs_dim]
        a_exec = self.buffer_r.actions[idx]
        s_cur = obs[:, -1, :] if self.current_frame and obs.ndim >= 3 else obs
        return s_cur, a_exec


def _episode_ends(dones):
    """返回每个 episode 的 exclusive end index。"""
    dones = np.asarray(dones).reshape(-1)
    ends = np.flatnonzero(dones).astype(np.int64) + 1
    if len(dones) and (len(ends) == 0 or ends[-1] != len(dones)):
        ends = np.append(ends, len(dones))
    return ends


def create_indices(episode_ends, sequence_length, episode_mask,
                   pad_before=0, pad_after=0):
    """为每个 episode 生成序列窗口索引。"""
    episode_ends = np.asarray(episode_ends)
    episode_mask = np.asarray(episode_mask, dtype=bool)
    sequence_length = int(sequence_length)
    pad_before = min(max(int(pad_before), 0), sequence_length - 1)
    pad_after = min(max(int(pad_after), 0), sequence_length - 1)
    indices = []
    for i, end_idx in enumerate(episode_ends):
        if not episode_mask[i]:
            continue
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        end_idx = int(end_idx)
        episode_length = end_idx - start_idx
        for idx in range(-pad_before, episode_length - sequence_length + pad_after + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            indices.append([buffer_start_idx, buffer_end_idx,
                            start_offset, sequence_length - end_offset])
    return np.asarray(indices, dtype=np.int64).reshape(-1, 4)


class SequenceSampler:
    """从 ExpertBuffer 快照按 indices 取序列，首尾边缘复制填充。"""

    def __init__(self, replay_buffer, sequence_length, pad_before=0, pad_after=0,
                 keys=None, key_first_k=None, episode_mask=None):
        data = replay_buffer.sequence_data()
        if keys is None:
            keys = list(data.keys())

        episode_ends = _episode_ends(data['dones'])
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        self.indices = create_indices(
            episode_ends, sequence_length, episode_mask, pad_before, pad_after,
        )
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.replay_buffer = data
        self.key_first_k = dict(key_first_k or {})

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = {}
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]

            if sample_start_idx > 0 or sample_end_idx < self.sequence_length:
                data = np.zeros(
                    (self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            else:
                data = sample
            result[key] = data
        return result

class ExpertSequenceDataset(Dataset):
    """按参考 diffusion policy 的时间约定生成观测和未来动作。"""

    def __init__(self, buffer, horizon, n_obs_steps,
                 pad_before=0, pad_after=0, episode_mask=None):
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        # 对应参考项目 predict_past_actions=False：只预测最新观测之后的动作。
        self.first_action_idx = self.n_obs_steps - 1
        self.sequence_length = self.first_action_idx + self.horizon
        self.sampler = SequenceSampler(
            buffer, sequence_length=self.sequence_length,
            pad_before=pad_before, pad_after=pad_after,
            keys=['obs', 'actions'],
            key_first_k={'obs': self.n_obs_steps},
            episode_mask=episode_mask,
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        return {
            'obs': torch.from_numpy(sample['obs'][:self.n_obs_steps].copy()),
            'action': torch.from_numpy(sample['actions'][
                self.first_action_idx:self.first_action_idx + self.horizon].copy()),
        }


class ExpertDataManager:
    """加载单帧 npz 专家数据，按 episode 划分训练/验证集。"""

    def __init__(self, capacity=100000):
        self.buffer = ExpertBuffer(capacity)
        self._raw_obs = None
        self._raw_actions = np.array([])
        self._raw_dones = np.array([])
        self._absorbing_state = False

    def load_data(self, data_path, raw_obs_dim=None, absorbing_state=False,
                  obs_normalizer=None):
        """先变换物理观测，再追加 absorbing bit / 自环，不更新统计量。"""
        with np.load(Path(data_path), allow_pickle=False) as data:
            raw_obs = np.asarray(data["states"], dtype=np.float32)
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            dones = np.asarray(data["dones"], dtype=np.float32).reshape(-1)
            terminated = truncated = None
            if absorbing_state:
                for key in ("terminated", "terminateds"):
                    if key in data:
                        terminated = np.asarray(data[key], dtype=bool).reshape(-1)
                        break
                for key in ("truncated", "truncations", "timeouts"):
                    if key in data:
                        mask = np.asarray(data[key], dtype=bool).reshape(-1)
                        if len(mask) != len(dones):
                            raise ValueError(f"专家 {key} 与 dones 长度不匹配")
                        truncated = mask if truncated is None else truncated | mask
        if raw_obs_dim is not None and raw_obs.shape[1] != int(raw_obs_dim):
            raise ValueError(f"states 维度不匹配：期望 {raw_obs_dim}，实际 {raw_obs.shape[1]}")
        if len(raw_obs) != len(raw_actions) or len(raw_obs) != len(dones):
            raise ValueError("专家 states / actions / dones 长度不匹配")
        if obs_normalizer is not None:
            raw_obs = np.asarray(obs_normalizer.normalize(raw_obs), dtype=np.float32)

        if absorbing_state:
            if terminated is not None:
                if len(terminated) != len(dones):
                    raise ValueError("专家 terminated 与 dones 长度不匹配")
                terminal_mask = terminated
            elif truncated is not None:
                terminal_mask = dones.astype(bool) & ~truncated
            else:
                warnings.warn(
                    "专家数据缺少 terminated/truncated metadata；"
                    "absorbing_state 将 legacy dones 视为真正 terminal（成功专家假设）。",
                    UserWarning, stacklevel=2,
                )
                terminal_mask = dones.astype(bool)

            boundaries = dones.astype(bool) | terminal_mask
            if truncated is not None:
                boundaries |= truncated
            raw_obs = np.concatenate([
                raw_obs, np.zeros((len(raw_obs), 1), dtype=np.float32),
            ], axis=-1)
            repeats = 1 + terminal_mask.astype(np.int64)
            loops = np.cumsum(repeats)[terminal_mask] - 1
            raw_obs = np.repeat(raw_obs, repeats, axis=0)
            raw_actions = np.repeat(raw_actions, repeats, axis=0)
            dones = np.repeat(boundaries.astype(np.float32), repeats)
            raw_obs[loops]     = 0
            raw_obs[loops, -1] = 1
            raw_actions[loops] = 0
            # ExpertBuffer 的 dones 表示序列边界；终止帧与自环归属同一回合。
            dones[loops - 1] = 0
            dones[loops] = 1

        self.buffer = ExpertBuffer(max(self.buffer.capacity, len(raw_obs)))
        self._raw_obs = raw_obs
        self._raw_actions = raw_actions
        self._raw_dones = dones
        self._absorbing_state = bool(absorbing_state)
        self.buffer.add_batch(raw_obs, raw_actions, dones)

    def trans_dataloader(self, split, batch_size, device=None, seed=0, *,
                         horizon=16, n_obs_steps=2, n_action_steps=8,
                         pad_before=None, pad_after=None,
                         num_workers=0, drop_last=False, obs_normalizer=None):
        """可选仅用训练回合拟合统计，再归一化全量数据并创建序列快照。"""
        episode_ends = _episode_ends(self._raw_dones)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(episode_ends))
        train_count = max(1, int(round(float(split) * len(episode_ends))))
        train_mask = np.zeros(len(episode_ends), dtype=bool)
        train_mask[order[:train_count]] = True

        if obs_normalizer is not None:
            episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
            train_frames = np.repeat(train_mask, episode_lengths)
            physical_obs = self._raw_obs
            if self._absorbing_state:
                train_frames &= physical_obs[:, -1] == 0
                physical_obs = physical_obs[:, :-1]
            obs_normalizer.update(physical_obs[train_frames])
            if self._absorbing_state:
                observations = self._raw_obs.copy()
                normal_mask = observations[:, -1] == 0
                observations[normal_mask, :-1] = obs_normalizer.normalize(physical_obs[normal_mask])
                observations[~normal_mask, :-1] = 0
            else:
                observations = obs_normalizer.normalize(physical_obs)
            self._raw_obs = np.asarray(observations, dtype=np.float32)
            self.buffer.clear()
            self.buffer.add_batch(self._raw_obs, self._raw_actions, self._raw_dones)

        if pad_before is None:
            pad_before = n_obs_steps - 1
        if pad_after is None:
            pad_after = n_action_steps - 1

        datasets = [
            ExpertSequenceDataset(self.buffer, horizon, n_obs_steps,
                                  pad_before, pad_after, mask)
            for mask in (train_mask, ~train_mask)
        ]
        pin_memory = device is not None and torch.device(device).type == 'cuda'
        train_loader = DataLoader(
            datasets[0], batch_size=batch_size, shuffle=True,
            drop_last=drop_last, num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            datasets[1], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return train_loader, val_loader
