"""旧 GAIL / off-policy 消融管线（MountainCar 链路）。

自 ``rl_utils.py`` 迁移：``train_sac_gail``（GASILfD / z-score / Trick1 消融开关）、
``train_off_policy_agent``、``warmup_replay_with_policy`` 及其专属池
``ReplayBuffer`` / ``TrajectoryBuffer`` 和 ``AsVecEnv``。

供 ``train/train_sac.py``、``train/train_sac_gail.py``、
``test_regression_compat.py`` 等旧脚本使用；
与 UR5e 定稿残差 SAC-GAIL（``rl_utils.py``）无关，新代码勿依赖。
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

from utils.rl_utils import (
    RunningMeanStd,
    rollout_eval,
    _vec_reset,
    _np,
    _final_info_for_env,
    _episode_was_success,
    _episode_slices,
    _concat_traj_sa,
    _sample_trajs_min_steps,
    _handle_eval_checkpoint,
    _load_gail_ckpt,
    orthogonal_init,
    _take_action,
)


def _add_policy_traj_for_disc(buffer_g, states, actions, info, env_idx,
                              oracle_filter=False, counters=None):
    """写入判别器负例池。oracle_filter 时成功轨迹不进 buffer_g。"""
    if oracle_filter and _episode_was_success(info, env_idx):
        if counters is not None:
            counters["skip"] = int(counters.get("skip", 0)) + 1
        return False
    buffer_g.add_trajectory(states, actions)
    if counters is not None:
        counters["add"] = int(counters.get("add", 0)) + 1
    return True


class AsVecEnv:
    """单环境 Gym → ``num_envs=1``，给 ``train_sac_gail`` 用。"""

    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        inner = getattr(env, "action_space", None)

        class _BatchedActionSpace:
            def sample(_self):
                a = inner.sample()
                return np.asarray(a, dtype=np.float32)[None, ...]

        self.action_space = _BatchedActionSpace()
        self.single_action_space = inner
        self.single_observation_space = getattr(env, "observation_space", None)
        self._ep_ret = 0.0
        self._ep_len = 0

    def _batch_obs(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        return x if x.ndim >= 3 else x[None, ...]

    def _pack_info(self, info):
        packed = {}
        for k, v in dict(info).items():
            if k in ("final_info", "final_observation"):
                continue
            arr = np.asarray(v)
            packed[k] = arr.reshape(1, *arr.shape) if arr.shape else np.array([v])
        packed.setdefault("success", np.array([bool(info.get("success", False))]))
        return packed

    def reset(self, seed=None, options=None, **kwargs):
        kw = dict(kwargs)
        if seed is not None:
            kw["seed"] = seed
        random_delta = None if options is None else options.get("random_delta")
        try:
            if random_delta is not None:
                obs, info = self.env.reset(random_delta=random_delta, **kw)
            else:
                obs, info = self.env.reset(**kw)
        except TypeError:
            obs, info = self.env.reset(**kw)
        self._ep_ret = 0.0
        self._ep_len = 0
        return self._batch_obs(obs), info

    def step(self, action):
        act = np.asarray(action, dtype=np.float32)
        if act.ndim == 2:
            act = act[0]
        obs, rew, term, trunc, info = self.env.step(act)
        self._ep_ret += float(rew)
        self._ep_len += 1
        packed = self._pack_info(info)
        done = bool(term or trunc)
        if done:
            packed["final_observation"] = np.empty(1, dtype=object)
            packed["final_observation"][0] = np.asarray(obs, dtype=np.float32)
            packed["final_info"] = {
                "episodic_return": np.array([self._ep_ret]),
                "episodic_length": np.array([self._ep_len]),
            }
            fi = info.get("final_info")
            if isinstance(fi, dict):
                packed["final_info"].update({k: np.array([v]) for k, v in fi.items()})
            obs, _ = self.env.reset()
            self._ep_ret = 0.0
            self._ep_len = 0
        return (
            self._batch_obs(obs),
            np.array([rew], dtype=np.float32),
            np.array([term], dtype=bool),
            np.array([trunc], dtype=bool),
            packed,
        )

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class ReplayBuffer:
    """numpy 预分配环形回放池。

    相比 deque 版本：
    - 采样通过 fancy indexing 一次性 memcpy，复杂度与容量无关（O(batch) 而非 O(n)）
    - 每条 transition 不再创建 Python tuple / ndarray 对象，消除 GC 扫描压力
    - 首次 add 时根据数据自动推断维度，保持向后兼容
    """

    def __init__(self, capacity, state_dim=None, action_dim=None):
        self.capacity = int(capacity)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._ready = False
        self._idx = 0
        self._size = 0

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
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity,) + state_shape, dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self._ready = True

    def add(self, state, action, reward, next_state, done):
        if not self._ready:
            self._build(state, action)
        i = self._idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.next_states[i] = next_state
        self.dones[i, 0] = float(done)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """一次写入 ``n_envs`` 条，避免 Python 逐条 ``add``。"""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        n = int(states.shape[0])
        if n == 0:
            return
        if not self._ready:
            self._build(states[0], actions[0])
        rewards = np.asarray(rewards, dtype=np.float32).reshape(n)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(n)
        i = self._idx
        cap = self.capacity
        if i + n <= cap:
            sl = slice(i, i + n)
            self.states[sl] = states
            self.actions[sl] = actions
            self.rewards[sl, 0] = rewards
            self.next_states[sl] = next_states
            self.dones[sl, 0] = dones
        else:
            n1 = cap - i
            self.states[i:] = states[:n1]
            self.actions[i:] = actions[:n1]
            self.rewards[i:, 0] = rewards[:n1]
            self.next_states[i:] = next_states[:n1]
            self.dones[i:, 0] = dones[:n1]
            n2 = n - n1
            self.states[:n2] = states[n1:]
            self.actions[:n2] = actions[n1:]
            self.rewards[:n2, 0] = rewards[n1:]
            self.next_states[:n2] = next_states[n1:]
            self.dones[:n2, 0] = dones[n1:]
        self._idx = (i + n) % cap
        self._size = min(self._size + n, cap)

    def sample(self, batch_size):
        if self._size == 0:
            raise ValueError("Replay buffer is empty, cannot sample.")
        indices = np.random.randint(0, self._size, size=batch_size)
        return (self.states[indices], self.actions[indices],
                self.rewards[indices], self.next_states[indices],
                self.dones[indices])

    def clear(self):
        self._idx = 0
        self._size = 0
        if self._ready:
            self.states.fill(0)
            self.actions.fill(0)
            self.rewards.fill(0)
            self.next_states.fill(0)
            self.dones.fill(0)

    def size(self):
        return self._size


class TrajectoryBuffer:
    """GAIL 策略占用：按完整轨迹存储 (s, a)。

    并行环境的 transition 在时间上交错，不能从环形 ReplayBuffer 的 done 还原轨迹。
    回合结束时 add_trajectory；判别器 sample_trajectories 抽出整条 τ。
    capacity 按步数计，超出时丢掉最旧轨迹。
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._trajs = collections.deque()
        self._n_steps = 0

    def add_trajectory(self, states, actions):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        n = int(states.shape[0])
        if n == 0:
            return
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"trajectory length mismatch: states={states.shape[0]} actions={actions.shape[0]}")
        self._trajs.append((states.copy(), actions.copy()))
        self._n_steps += n
        while self._n_steps > self.capacity and len(self._trajs) > 1:
            old_s, _ = self._trajs.popleft()
            self._n_steps -= int(old_s.shape[0])

    def n_trajectories(self):
        return len(self._trajs)

    def sample_trajectories(self, min_steps=1):
        """抽完整轨迹直到 (s,a) 总数 ≥ min_steps（短局会多抽几条）。"""
        n = self.n_trajectories()
        if n == 0:
            raise ValueError("Trajectory buffer is empty, cannot sample.")
        return _sample_trajs_min_steps(n, lambda i: self._trajs[i], min_steps)

    def clear(self):
        self._trajs.clear()
        self._n_steps = 0

    def size(self):
        return self._n_steps

 

def _policy_step(agent, obs, running_ms=None, deterministic=False, base_only=False):
    """返回 (replay_action, executed_action)。

    残差模式：replay 为 π_θ，executed 为 clip(π_H+α·π_θ(s,â))；否则两者相同。
    ``base_only``：DAWN 预热，执行 π_H、回放 ã=0。
    """
    executed = _take_action(
        agent, obs, running_ms=running_ms, deterministic=deterministic, base_only=base_only)
    if hasattr(agent, "replay_action"):
        try:
            stored = np.array(agent.replay_action(), dtype=np.float32, copy=True)
        except RuntimeError:
            stored = executed
    else:
        stored = executed
    return stored, executed


def _rollout_eval_vec(env, agent, n_episodes=20, running_ms=None,
                      deterministic=True, seed_offset=100000):
    """并行评估：env 为 num_envs>1 的 AssembleVecEnv（squeeze=False）。

    各环境 reset 时已按 env 独立抖动视觉孔位，无需逐 env 设种子。
    为避免「快回合被重复采样、慢回合漏采」的偏差，每轮每个环境只取它的第一条
    完整回合；集满 n_episodes 条后返回与单环境版一致的指标字典。
    建议令 n_episodes == num_envs（恰好一整轮，无偏）。
    """
    n = int(env.num_envs)
    obs, _ = env.reset(seed=seed_offset)
    ep_ret = np.zeros(n, dtype=np.float64)
    ep_len = np.zeros(n, dtype=np.int64)
    ep_peak = np.zeros(n, dtype=np.float64)
    harvested = np.zeros(n, dtype=bool)

    returns, lengths, successes, peak_forces, final_states = [], [], [], [], []
    completed = 0
    while completed < n_episodes:
        action = _take_action(agent, obs, running_ms=running_ms, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        term = np.asarray(terminated).reshape(-1).astype(bool)
        trunc = np.asarray(truncated).reshape(-1).astype(bool)
        done = term | trunc
        force = np.asarray(info.get("force", np.zeros((n, 6))), dtype=np.float32).reshape(n, -1)
        succ = np.asarray(info.get("success", np.zeros(n)), dtype=bool).reshape(-1)
        state = np.asarray(info.get("state", np.full(n, -1))).reshape(-1)

        ep_ret += reward
        ep_len += 1
        ep_peak = np.maximum(ep_peak, np.linalg.norm(force[:, :6], axis=-1))

        for j in np.flatnonzero(done):
            if completed >= n_episodes:
                break
            if harvested[j]:
                continue  # 本回合已取过，等下一轮
            harvested[j] = True
            returns.append(float(ep_ret[j]))
            lengths.append(int(ep_len[j]))
            peak_forces.append(float(ep_peak[j]))
            successes.append(1.0 if (term[j] and succ[j]) else 0.0)
            final_states.append(int(state[j]))
            completed += 1
        ep_ret[done] = 0.0
        ep_len[done] = 0
        ep_peak[done] = 0.0
        if harvested.all():
            harvested[:] = False  # 一整轮取完，允许进入下一轮（n_episodes > n 时）

    states, counts = np.unique(final_states, return_counts=True)
    return {
        'success_rate':    float(np.mean(successes)),
        'return_mean':     float(np.mean(returns)),
        'return_std':      float(np.std(returns)),
        'length_mean':     float(np.mean(lengths)),
        'peak_force_mean': float(np.mean(peak_forces)),
        'state_dist':      {int(s): int(c) for s, c in zip(states, counts)},
        'n_episodes':      int(len(successes)),
    }


def train_off_policy_agent(env, eval_env, agent, total_timesteps, train_freq, gradient_steps,
                           replay_buffer, learning_starts, batch_size, seed=1,
                           eval_interval=5000, eval_episodes=10,
                           is_save_model=True, is_draw=True, normalized_observation=True,
                           save_model_dir: Path = None, log_dir: Path = None):
    """通用 off-policy 训练循环（适配 gym 标准环境与 AssembleMuJoCoEnv）。

    依赖 EpisodeStatsWrapper 提供的 info['final_info'] = {episodic_return, episodic_length}，
    请在传入 env/eval_env 前用 EpisodeStatsWrapper 包装。
    """
    return_list = [[], []]  # [returns, global_steps]
    global_step = 0
    episodic_count = 0
    recent_avg_window = 10
    best_return = -np.inf
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{exp_name}__{seed}__{int(time.time())}"
    last_record_time = time.time()
    last_record_step = 0
    writer = SummaryWriter(log_dir / run_name)
    if normalized_observation:
        obs_normalizer = RunningMeanStd(env.observation_space.shape)

    # 初始化环境
    observation, _ = env.reset()
    if normalized_observation:
        obs_normalizer.update(np.array([observation]))

    # 训练循环
    for i in range(10):
        with tqdm(total=int(total_timesteps / 10), desc=f"Iteration:{i:d}", dynamic_ncols=True, mininterval=0.5) as pbar:
            while pbar.n < pbar.total:
                # 采样
                if global_step < learning_starts:
                    action = env.action_space.sample()
                else:
                    obs_in = obs_normalizer.normalize(observation) if normalized_observation else observation
                    action = agent.take_action(obs_in)

                next_observation, reward, terminated, truncated, info = env.step(action)
                if normalized_observation:
                    obs_normalizer.update(np.array([next_observation]))

                # 注意：dones 只存 terminated（真正终止）。
                # truncated 是时间截断，critic 需要对它 bootstrap，否则价值估计会有偏。
                replay_buffer.add(observation, action, reward, next_observation, terminated)
                observation = next_observation

                # 回合结束（依赖 EpisodeStatsWrapper 注入的 final_info）
                if 'final_info' in info:
                    episodic_return = info['final_info']['episodic_return']
                    episodic_length = info['final_info']['episodic_length']
                    episodic_count += 1
                    return_list[0].append(episodic_return)
                    return_list[1].append(global_step)

                    # episode 重置
                    observation, _ = env.reset()
                    if normalized_observation:
                        obs_normalizer.update(np.array([observation]))
                    if hasattr(agent, 'reset_noise'):
                        agent.reset_noise()  # 重置探索噪声（gSDE）

                    avg_return = np.mean(return_list[0][-recent_avg_window:])

                    # AssembleMuJoCoEnv 才有 'state' 字段（状态机：0R/1S/2I/3F/4S）
                    trans_state = info.get('state', None)

                    if is_draw:
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
                        writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                        if trans_state is not None:
                            writer.add_scalar("charts/trans_state", trans_state, global_step)

                    if is_save_model and save_model_dir and avg_return > best_return:
                        best_return = avg_return
                        agent.save_model(save_model_dir / "best_model")
                        best_info = {
                            'episode': episodic_count,
                            'global_step': int(global_step),
                            'best_return': float(best_return),
                            'avg_return': float(avg_return),
                        }
                        if normalized_observation:
                            best_info['obs_normalizer'] = obs_normalizer.state_dict()
                        json_path = save_model_dir / "best_model" / "info.json"
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(best_info, f, indent=2, ensure_ascii=False)

                    pbar.update(episodic_length)
                    pbar.set_postfix({
                        'episode': episodic_count,
                        'global_step': f"{global_step:06d}",
                        'return': f"{episodic_return:.3f}",
                        'best_return': f"{best_return:.3f}",
                    })

                # 训练
                if (replay_buffer.size() > learning_starts) and (global_step % train_freq == 0):
                    avg_info = {}
                    for _ in range(gradient_steps):
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        if normalized_observation:
                            b_s = obs_normalizer.normalize(b_s)
                            b_ns = obs_normalizer.normalize(b_ns)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns,
                                           'rewards': b_r, 'dones': b_d}
                        train_info = agent.update(transition_dict)  # 改名避免覆盖 env.step 的 info
                        for k, v in train_info.items():
                            avg_info[k] = avg_info.get(k, 0.0) + v
                    for k in avg_info:
                        avg_info[k] /= gradient_steps

                    if global_step % 100 == 0 and is_draw:
                        current_time = time.time()
                        delta_time = current_time - last_record_time
                        delta_steps = global_step - last_record_step
                        if delta_time > 0:
                            sps = int(delta_steps / delta_time)
                            writer.add_scalar("charts/SPS", sps, global_step)
                        last_record_time = current_time
                        last_record_step = global_step
                        for k, v in avg_info.items():
                            writer.add_scalar(f"losses/{k}", v, global_step)

                global_step += 1

                # 周期评估
                if (eval_env is not None
                        and global_step > learning_starts
                        and global_step % eval_interval == 0):
                    stats = rollout_eval(eval_env, agent, n_episodes=eval_episodes,
                                         running_ms=obs_normalizer if normalized_observation else None)
                    if is_draw:
                        writer.add_scalar("eval/success_rate", stats['success_rate'], global_step)
                        writer.add_scalar("eval/return_mean", stats['return_mean'], global_step)
                        writer.add_scalar("eval/peak_force_mean", stats['peak_force_mean'], global_step)
                    pbar.write(f"[eval@{global_step}] success={stats['success_rate']:.1%} "
                               f"ret={stats['return_mean']:.1f}±{stats['return_std']:.1f} "
                               f"peakF={stats['peak_force_mean']:.1f}N states={stats['state_dist']}")

        if is_save_model and save_model_dir:
            agent.save_model(save_model_dir / "final_model")
            final_model_info = {
                'best_return': float(best_return),
                'return': float(return_list[0][-1]) if return_list[0] else None,
                'global_step': int(global_step),
                'episode': int(episodic_count),
            }
            if normalized_observation:
                final_model_info['obs_normalizer'] = obs_normalizer.state_dict()
            json_path = save_model_dir / "final_model" / "info.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_model_info, f, indent=2, ensure_ascii=False)

        # 终检
        if eval_env is not None:
            final_stats = rollout_eval(eval_env, agent, n_episodes=max(30, eval_episodes),
                                       running_ms=obs_normalizer if normalized_observation else None,
                                       seed_offset=200000)
            print(f"[final eval] success={final_stats['success_rate']:.1%} "
                  f"ret={final_stats['return_mean']:.1f}±{final_stats['return_std']:.1f}")
            eval_env.close()

    writer.close()
    return np.asarray(return_list, dtype=np.float32)


def _gail_rms_mu_std(gail_r_rms):
    """标量 (μ, σ)；σ = sqrt(var + eps)，不含 std_floor。"""
    mu = float(np.asarray(gail_r_rms.mean, dtype=np.float64).reshape(-1)[0])
    var = float(np.asarray(gail_r_rms.var, dtype=np.float64).reshape(-1)[0])
    sig = float(np.sqrt(var + float(gail_r_rms.epsilon)))
    return mu, sig


def _gail_reward_normalize(r_arr, gail_r_rms, is_update=True,
                           z_clip=2.0, std_floor=0.2):
    """GAIL 奖励 z-score：σ 下限 + 紧 z 裁剪。

    z = clip( (r - μ) / max(σ, std_floor), -z_clip, z_clip )

    不用 ``RunningMeanStd.normalize``（默认 clip=±50）。D 饱和后 σ→0 时，
    ±50 会把逐步奖励放到几十、回合和放到上千。
    """
    z_clip = None if z_clip is None else float(z_clip)
    std_floor = max(0.0, float(std_floor))

    if is_update:
        gail_r_rms.update(r_arr.reshape(-1, 1) if torch.is_tensor(r_arr)
                          else np.asarray(r_arr, dtype=np.float32).reshape(-1, 1))
    mu, sig = _gail_rms_mu_std(gail_r_rms)
    denom = max(sig, std_floor) if std_floor > 0.0 else sig

    if torch.is_tensor(r_arr):
        orig_shape = r_arr.shape
        mean_t = torch.as_tensor(mu, dtype=r_arr.dtype, device=r_arr.device)
        denom_t = torch.as_tensor(denom, dtype=r_arr.dtype, device=r_arr.device)
        z = (r_arr - mean_t) / denom_t
        if z_clip is not None:
            z = torch.clamp(z, -z_clip, z_clip)
        return z.reshape(orig_shape)

    r_arr = np.asarray(r_arr, dtype=np.float32)
    orig_shape = r_arr.shape
    z = (r_arr.astype(np.float64) - mu) / denom
    if z_clip is not None:
        z = np.clip(z, -z_clip, z_clip)
    return z.astype(np.float32).reshape(orig_shape)


def _finite_mean(values, mask):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if x.shape[0] != m.shape[0]:
        n = min(x.shape[0], m.shape[0])
        x, m = x[:n], m[:n]
    m = m & np.isfinite(x)
    if not np.any(m):
        return None
    return float(x[m].mean())


def warmup_replay_with_policy(env, agent, buffer_r, n_steps, running_ms=None,
                              buffer_g=None, random_delta=None, oracle_filter=False,
                              counters=None, seed=None):
    """用当前策略（BC / base policy）采 ``n_steps`` 条转移写入 ``buffer_r``。

    不更新网络。存环境逐步奖励；GAIL 分在训练时由判别器现算。
    完整回合额外写入 ``buffer_g``，避免判别器开局把 Trick1 专家当策略占用。
    ``oracle_filter``：成功轨迹不进 ``buffer_g``。
    ``seed``：首次 ``reset`` 播种；worker i 为 seed+i。None 则不播（沿用已有 rng）。
    残差模式（DAWN）：只用冻住的 π_H 采集，buffer_r 写入 ã=0，不当随机残差。
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        return 0
    if getattr(env, "num_envs", None) is None:
        env = AsVecEnv(env)
    n_envs = int(getattr(env, "num_envs", 1))
    reset_options = {"random_delta": random_delta} if random_delta is not None else None
    observation, _ = _vec_reset(env, seed=seed, options=reset_options)
    observation = _np(observation).astype(np.float32)
    _sas = getattr(env, "single_action_space", None)
    act_dim = int(getattr(_sas, "shape", (6,))[0]) if _sas is not None else 6
    obs_tail = tuple(observation.shape[1:])
    ep_cap = 512
    ep_states = np.zeros((n_envs, ep_cap) + obs_tail, dtype=np.float32)
    ep_actions = np.zeros((n_envs, ep_cap, act_dim), dtype=np.float32)
    ep_len = np.zeros(n_envs, dtype=np.int64)
    collected = 0
    n_traj = 0
    base_only = getattr(agent, "base_actor", None) is not None
    if hasattr(agent, "reset_noise"):
        agent.reset_noise(batch_size=n_envs)
    warmup_desc = "warmup(base policy)" if base_only else "warmup(policy)"
    with tqdm(total=n_steps, desc=warmup_desc, dynamic_ncols=True, mininterval=0.5) as pbar:
        while collected < n_steps:
            action_store, action = _policy_step(
                agent, observation, running_ms=running_ms,
                deterministic=True if base_only else False,
                base_only=base_only)
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            next_observation = _np(next_observation).astype(np.float32)
            env_reward = _np(env_reward).reshape(n_envs).astype(np.float32)
            terminated = _np(terminated).reshape(n_envs).astype(bool)
            truncated = _np(truncated).reshape(n_envs).astype(bool)
            if running_ms:
                running_ms.update(next_observation)
            next_states = next_observation
            dones = terminated | truncated
            final_obs_arr = info.get("final_observation", None) if np.any(dones) else None
            if final_obs_arr is not None:
                next_states = next_observation.copy()
                for j in np.flatnonzero(dones):
                    fo_j = final_obs_arr[j]
                    if fo_j is not None:
                        next_states[j] = fo_j
            buffer_r.add_batch(observation, action_store, env_reward, next_states, terminated)
            t = ep_len
            if int(t.max()) + 1 >= ep_cap:
                new_cap = max(int(t.max()) + 2, ep_cap * 2)
                ns = np.zeros((n_envs, new_cap) + obs_tail, dtype=np.float32)
                na = np.zeros((n_envs, new_cap, act_dim), dtype=np.float32)
                ns[:, :ep_cap] = ep_states
                na[:, :ep_cap] = ep_actions
                ep_states, ep_actions, ep_cap = ns, na, new_cap
            rows = np.arange(n_envs)
            ep_states[rows, t] = observation
            ep_actions[rows, t] = action
            ep_len += 1
            observation = next_observation
            if np.any(dones):
                if buffer_g is not None:
                    for j in np.flatnonzero(dones):
                        L = int(ep_len[j])
                        if _add_policy_traj_for_disc(
                                buffer_g, ep_states[j, :L], ep_actions[j, :L],
                                info, int(j), oracle_filter=oracle_filter,
                                counters=counters):
                            n_traj += 1
                        ep_len[j] = 0
                else:
                    ep_len[np.flatnonzero(dones)] = 0
            collected += n_envs
            pbar.update(n_envs)
    print(
        f"[warmup] buffer_r={buffer_r.size()} steps from "
        f"{'base policy (ã=0)' if base_only else 'current policy'} "
        f"(target {n_steps}); buffer_g += {n_traj} trajs."
    )
    return int(buffer_r.size())


def train_sac_gail(env, eval_env, agent, disc, total_timesteps, train_freq, gradient_steps,
                   buffer_r, buffer_g, buffer_e, learning_starts, batch_size,
                   seed=1, eval_interval=5000, eval_episodes=10,
                   is_save_model=True, is_draw=True, running_ms=None,
                   save_model_dir: Path = None, log_dir: Path = None,
                   env_reward_weight: float = 0.0,
                   disc_update_freq: int = 1,
                   gail_reward_runnorm: bool = False,
                   gail_reward_z_clip: float = 2.0,
                   gail_reward_std_floor: float = 0.2,
                   gail_reward_coef: float = 1.0,
                   random_delta: np.ndarray = None,
                   run_name: str = None,
                   use_gasilf: bool = False,
                   gasilf_success_threshold: float = 90.0,
                   gasilf_add_cap: int = 200,
                   warmup_steps: int = 0,
                   bc_coef: float = 0.0,
                   bc_onpol_coef: float = 0.0,
                   oracle_filter: bool = False,
                   expert_suffix_frac: float = 0.0,
                   expert_suffix_ratio: float = 0.0):
    """GAIL + SAC。单环境自动用 AsVecEnv 包成 num_envs=1。

    env 为 ``gym.vector.VectorEnv`` / ``AsVecEnv``：每次 ``step`` 推进
    ``n_envs`` 个环境。

    ``gail_reward_runnorm`` 为 True 时：z = clip((r-μ)/max(σ, std_floor), ±z_clip)。
    混合：``r = gail_reward_coef * z + env_reward_weight * r_env``。
    TensorBoard：``episodic_gail_return`` 是 coef×z 的回合和（SAC 实际用的），
    ``episodic_gail_return_raw`` 是判别器逐步奖励×scale、乘 coef 之前的回合和
    （``neglogd`` 为 −log(D)，``logit`` 为 clip(-ℓ,±c)）。

    ``use_gasilf``：回合环境回报 ``> gasilf_success_threshold`` 时，把该轨迹
    追加进 ``buffer_e``（label=0），最多 ``gasilf_add_cap`` 条。UR5e 默认关。

    ``warmup_steps``：开训前用当前策略（通常是 BC）往 ``buffer_r`` 写入转移，
    并令 ``learning_starts=0``，避免原先用 ``action_space.sample()`` 的随机填充
    把 Critic 锚到乱走上。不计入 ``total_timesteps``。
    训练 VectorEnv 仅在第一次 ``reset`` 传入 ``seed``（worker i 为 seed+i）；
    之后 autoreset / 再 reset 不重新播种，沿用同一条 np_random。

    ``bc_coef``：actor 附加 ``β MSE(μ(s_E), a_E)``，``s_E,a_E`` 从 ``buffer_e``
    与 SAC 同 batch 抽取。0=关闭。克隆后 MSE≈0.02、|Q|~10 时 β=50 使 BC 项与
    约 1 个 Q 同阶，避免随机 critic 把均值拉开。

    ``bc_onpol_coef``：在 replay 状态（近似 ρ_π）上 ``β MSE(μ_θ(s), μ_BC(s))``，
    需先 ``agent.freeze_actor_as_bc()``。与 ``bc_coef`` 独立，隔离时不要同时开。

    ``oracle_filter``：成功策略轨迹不写入 ``buffer_g``（判别器负例），避免插入
    被标成 policy。不写入 ``buffer_e``（那是 GASILfD）。

    ``expert_suffix_frac`` / ``expert_suffix_ratio``：判别器和 ρ_E BC 从专家轨迹
    末段过采样插入占用。0=均匀抽整条。

    Residual RL（``agent.base_actor`` 非空）：``take_action`` 返回
    执行动作 clip(π_H+α·π_θ(s,â))，``replay_action`` 写入 ``buffer_r``；
    ``buffer_g`` 仍记执行占用。BeTSAC 用环境奖励（c=0）；BeTAIL 的 D 与
    GAIL 奖励打在执行动作 â+αã 上，不打在残差 ã 上。

    残差预热（DAWN）：只用 π_H 采集，buffer_r 写 ã=0，不显式预训 Critic。
    """
    if getattr(env, "num_envs", None) is None:
        env = AsVecEnv(env)
    n_envs = int(getattr(env, "num_envs", 1))
    reset_options = {"random_delta": random_delta} if random_delta is not None else None

    # ==================== 1. 初始化 ====================
    return_list = []
    gail_loss = []
    global_step = 0
    episodic_count = 0
    recent_avg_window = 10
    best_success = 0.0
    disc_info = {}
    logged_d_pol = None

    if not run_name:
        exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{exp_name}__{seed}__{int(time.time())}"
    writer = SummaryWriter(log_dir / run_name)
    gasilf_added = 0
    use_gasilf = bool(use_gasilf)
    gasilf_add_cap = max(0, int(gasilf_add_cap))
    gasilf_success_threshold = float(gasilf_success_threshold)
    last_record_time = time.time()
    last_record_step = 0

    gail_r_rms = RunningMeanStd(shape=()) if gail_reward_runnorm else None
    use_gail_r = abs(float(gail_reward_coef)) > 0.0
    if not use_gail_r:
        print("[train] gail_reward_coef=0: skip discriminator updates and GAIL reward")
    bc_coef = float(bc_coef)
    bc_onpol_coef = float(bc_onpol_coef)
    oracle_filter = bool(oracle_filter)
    expert_suffix_frac = float(expert_suffix_frac)
    expert_suffix_ratio = float(expert_suffix_ratio)
    oracle_g_stats = {"add": 0, "skip": 0}

    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0:
        warmup_replay_with_policy(
            env, agent, buffer_r, warmup_steps,
            running_ms=running_ms, buffer_g=buffer_g, random_delta=random_delta,
            oracle_filter=oracle_filter,
            counters=oracle_g_stats,
            seed=seed,
        )
        # 预热已用策略填充回放；主循环不再用随机动作冲掉锚点
        learning_starts = 0
        if is_draw:
            writer.add_scalar("charts/warmup_buffer_r", float(buffer_r.size()), 0)
        train_reset_seed = None
    else:
        train_reset_seed = seed

    observation, _ = _vec_reset(env, seed=train_reset_seed, options=reset_options)
    observation = _np(observation).astype(np.float32)
    # SB3 collect_rollouts：每段采集开头 reset_noise(n_envs)；回合结束不重抽 gSDE。
    if hasattr(agent, "reset_noise"):
        agent.reset_noise(batch_size=int(getattr(env, "num_envs", 1)))
    episodic_return = np.zeros(n_envs, dtype=np.float64)
    episodic_env_return = np.zeros(n_envs, dtype=np.float64)
    episodic_gail_return = np.zeros(n_envs, dtype=np.float64)
    episodic_gail_return_raw = np.zeros(n_envs, dtype=np.float64)
    episodic_length = np.zeros(n_envs, dtype=np.int64)
    obs_tail = tuple(observation.shape[1:])
    _sas = getattr(env, "single_action_space", None)
    act_dim = int(getattr(_sas, "shape", (6,))[0]) if _sas is not None else 6
    ep_cap = 512
    ep_states = np.zeros((n_envs, ep_cap) + obs_tail, dtype=np.float32)
    ep_actions = np.zeros((n_envs, ep_cap, act_dim), dtype=np.float32)

    def _grow_ep_buf(need):
        nonlocal ep_cap, ep_states, ep_actions
        if need < ep_cap:
            return
        new_cap = max(int(need) + 1, ep_cap * 2)
        ns = np.zeros((n_envs, new_cap) + obs_tail, dtype=np.float32)
        na = np.zeros((n_envs, new_cap, act_dim), dtype=np.float32)
        ns[:, :ep_cap] = ep_states
        na[:, :ep_cap] = ep_actions
        ep_states, ep_actions, ep_cap = ns, na, new_cap

    # ==================== 2. 主训练循环 ====================
    steps_per_iter = total_timesteps // 10
    for i in range(10):
        with tqdm(total=steps_per_iter, desc=f"Iteration:{i:d}",
                  dynamic_ncols=True, mininterval=0.5) as pbar:
            while pbar.n < pbar.total:
                # ---------- 2.1 采样 ----------
                # 残差模式必须走 π_H+π_θ，不能 action_space.sample()（那会把满幅度随机动作当残差写入）。
                use_random_fill = (
                    global_step < learning_starts and buffer_r.size() == 0
                    and getattr(agent, "base_actor", None) is None
                )
                if use_random_fill:
                    if hasattr(env, "action_space") and hasattr(env.action_space, "sample"):
                        action = env.action_space.sample()
                    else:
                        action = np.random.uniform(-1.0, 1.0, size=observation.shape[:1] + (act_dim,)).astype(np.float32)
                    action = np.asarray(action, dtype=np.float32)
                    action_store = action
                else:
                    action_store, action = _policy_step(
                        agent, observation, running_ms=running_ms, deterministic=False)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                next_observation = _np(next_observation).astype(np.float32)
                env_reward = _np(env_reward).reshape(n_envs).astype(np.float32)
                terminated = _np(terminated).reshape(n_envs).astype(bool)
                truncated = _np(truncated).reshape(n_envs).astype(bool)
                if running_ms:
                    running_ms.update(next_observation)

                # ---------- 2.2 存入回放池 ----------
                # VectorEnv 在 episode 结束时会 reset，next_observation[j] 是新 episode
                # 首帧。done 时用 final_observation 作为 next_state。
                next_states = next_observation
                dones = terminated | truncated
                final_obs_arr = info.get('final_observation', None) if np.any(dones) else None
                if final_obs_arr is not None:
                    next_states = next_observation.copy()
                    for j in np.flatnonzero(dones):
                        fo_j = final_obs_arr[j]
                        if fo_j is not None:
                            next_states[j] = fo_j
                buffer_r.add_batch(observation, action_store, env_reward, next_states, terminated)

                t = episodic_length
                if t.size:
                    _grow_ep_buf(int(t.max()) + 1)
                    rows = np.arange(n_envs)
                    ep_states[rows, t] = observation
                    ep_actions[rows, t] = action

                observation = next_observation
                episodic_env_return += env_reward
                episodic_length += 1

                # ---------- 2.3 / 2.4 Episode 结束：GAIL 监控奖励只在 done 时算一次 ----------
                if np.any(dones):
                    final_info = info.get("final_info", {}) if isinstance(info, dict) else {}
                    ep_lengths = np.asarray(
                        final_info.get("episodic_length", episodic_length), dtype=np.int64
                    ).reshape(n_envs)
                    term_nan = np.asarray(
                        info.get("term_nan", np.zeros(n_envs, dtype=bool)), dtype=bool
                    ).reshape(-1)
                    if term_nan.shape[0] < n_envs:
                        term_nan = np.pad(term_nan, (0, n_envs - term_nan.shape[0]))
                    done_idx = np.where(dones)[0]

                    chunks_s, chunks_a, done_lens = [], [], []
                    for j in done_idx:
                        L = int(episodic_length[j])
                        chunks_s.append(ep_states[j, :L])
                        chunks_a.append(ep_actions[j, :L])
                        done_lens.append(L)
                        if use_gail_r:
                            _add_policy_traj_for_disc(
                                buffer_g, ep_states[j, :L], ep_actions[j, :L],
                                info, int(j), oracle_filter=oracle_filter,
                                counters=oracle_g_stats)
                        episodic_count += 1

                    S = np.concatenate(chunks_s, axis=0)
                    if use_gail_r:
                        A = np.concatenate(chunks_a, axis=0)
                        if running_ms:
                            S = running_ms.normalize(S)
                        r_all = np.asarray(disc.predict_rewards(S, A), dtype=np.float32).reshape(-1)
                        if gail_r_rms is not None:
                            r_all_n = _gail_reward_normalize(
                                r_all, gail_r_rms, is_update=False,
                                z_clip=gail_reward_z_clip, std_floor=gail_reward_std_floor,
                            ).reshape(-1)
                        else:
                            r_all_n = r_all
                    else:
                        r_all = np.zeros(int(S.shape[0]), dtype=np.float32)
                        r_all_n = r_all
                    off = 0
                    done_returns = np.empty(len(done_idx), dtype=np.float64)
                    done_env_returns = episodic_env_return[done_idx].copy()
                    done_gail_returns = np.empty(len(done_idx), dtype=np.float64)
                    done_gail_returns_raw = np.empty(len(done_idx), dtype=np.float64)
                    for k, j in enumerate(done_idx):
                        L = done_lens[k]
                        r_n = r_all_n[off:off + L]
                        r_raw = r_all[off:off + L]
                        off += L
                        gail_sum = float(gail_reward_coef) * float(r_n.sum())
                        gail_raw_sum = float(r_raw.sum())
                        env_sum = float(done_env_returns[k])
                        mixed = gail_sum + env_reward_weight * env_sum
                        done_returns[k] = mixed
                        done_gail_returns[k] = gail_sum
                        done_gail_returns_raw[k] = gail_raw_sum
                        return_list.append(mixed)
                        if (use_gasilf
                                and gasilf_added < gasilf_add_cap
                                and env_sum > gasilf_success_threshold):
                            buffer_e.add_trajectory(chunks_s[k], chunks_a[k])
                            gasilf_added += 1
                        episodic_return[j] = 0.0
                        episodic_env_return[j] = 0.0
                        episodic_gail_return[j] = 0.0
                        episodic_gail_return_raw[j] = 0.0
                        episodic_length[j] = 0

                    if is_draw:
                        tb_step = global_step + n_envs
                        done_mask = dones
                        task_mask = done_mask & ~term_nan[:n_envs]
                        avg_return = np.mean(return_list[-recent_avg_window:]) if return_list else 0.0
                        ep_x = int(episodic_count)
                        writer.add_scalar("charts/episodic_return", float(np.mean(done_returns)), ep_x)
                        writer.add_scalar("charts/avg_episodic_return", avg_return, ep_x)
                        writer.add_scalar("charts/episodic_length", float(ep_lengths[done_mask].mean()), ep_x)
                        writer.add_scalar("charts/episodic_env_return", float(np.mean(done_env_returns)), ep_x)
                        writer.add_scalar("charts/episodic_gail_return", float(np.mean(done_gail_returns)), ep_x)
                        writer.add_scalar(
                            "charts/episodic_gail_return_raw",
                            float(np.mean(done_gail_returns_raw)), ep_x)
                        done_lens_f = np.asarray(done_lens, dtype=np.float64)
                        writer.add_scalar(
                            "charts/expert_reward",
                            float(np.mean(done_gail_returns_raw / np.maximum(done_lens_f, 1.0))),
                            ep_x)
                        writer.add_scalar("gasilf/total_added_count", int(gasilf_added), ep_x)
                        if oracle_filter:
                            _n_g = int(oracle_g_stats["add"] + oracle_g_stats["skip"])
                            writer.add_scalar(
                                "charts/oracle_filter_skip_frac",
                                float(oracle_g_stats["skip"] / max(_n_g, 1)), ep_x)
                            writer.add_scalar(
                                "charts/oracle_filter_skipped",
                                float(oracle_g_stats["skip"]), ep_x)
                        if use_gasilf:
                            writer.add_scalar("gasilf/expert_buffer_size", int(buffer_e.size()), ep_x)
                            writer.add_scalar(
                                "gasilf/expert_n_traj", int(buffer_e.n_trajectories()), ep_x)
                        if gail_r_rms is not None:
                            mu, sig = _gail_rms_mu_std(gail_r_rms)
                            writer.add_scalar("charts/gail_r_rms_mean", mu, ep_x)
                            writer.add_scalar("charts/gail_r_rms_std", sig, ep_x)
                            writer.add_scalar(
                                "charts/gail_r_std_used",
                                max(sig, float(gail_reward_std_floor)), ep_x)
                        writer.add_scalar("charts/done_frac", float(done_mask.mean()), tb_step)
                        for name in ("fail", "success", "time_out"):
                            key = f"term_{name}"
                            if key in info:
                                writer.add_scalar(
                                    f"charts/term_{name}_frac",
                                    float(np.asarray(info[key]).reshape(-1)[:n_envs][done_mask].mean()),
                                    tb_step,
                                )
                        depth_v = _finite_mean(info.get("depth", final_info.get("depth", [])), task_mask)
                        xy_v = _finite_mean(info.get("position_error_xy", final_info.get("position_error_xy", [])), task_mask)
                        yaw_v = _finite_mean(info.get("yaw_error", final_info.get("yaw_error", [])), task_mask)
                        if depth_v is not None:
                            writer.add_scalar("tasks/depth (m)", depth_v, tb_step)
                        if xy_v is not None:
                            writer.add_scalar("tasks/position_error_xy (m)", xy_v, tb_step)
                        if yaw_v is not None:
                            writer.add_scalar("tasks/yaw_error (deg)", yaw_v, tb_step)

                    d_exp = disc_info.get('expert_value', None) if disc_info else None
                    d_pol = logged_d_pol if logged_d_pol is not None else (
                        disc_info.get('policy_value') if disc_info else None)
                    pbar.set_postfix({
                        'episode': episodic_count,
                        'global_step': f"{global_step:06d}",
                        'ep_len': f"{float(ep_lengths[dones].mean()):.1f}",
                        'nan': f"{float(term_nan[:n_envs][dones].mean()):.2f}",
                        'D_exp': f"{d_exp:.3f}" if d_exp is not None else "-",
                        'D_pol': f"{d_pol:.3f}" if d_pol is not None else "-",
                        'gail_loss': f"{np.mean(gail_loss[-10:]):.3f}" if gail_loss else "0.000",
                        'gasilf': gasilf_added,
                    })

                # ---------- 2.5 策略与判别器训练 ----------
                # n_envs=14、train_freq=64 时 +14 对不齐 64，实际约每 448 条转移训一轮。
                if (buffer_r.size() > learning_starts) and (global_step % train_freq == 0):
                    avg_info = {}
                    disc_update_counter = 0
                    disc_info = {}
                    last_gi = gradient_steps - 1
                    for gi in range(gradient_steps):
                        log_info = gi == last_gi
                        if use_gail_r:
                            disc_update_counter += 1
                            if disc_update_counter >= disc_update_freq:
                                disc_update_counter = 0
                                disc_log = (last_gi - gi) < disc_update_freq
                                try:
                                    _dinfo = disc.update(
                                        buffer_e, buffer_g, running_ms, log_info=disc_log,
                                        expert_suffix_frac=expert_suffix_frac,
                                        expert_suffix_ratio=expert_suffix_ratio,
                                    )
                                except TypeError:
                                    _dinfo = disc.update(buffer_e, buffer_g, running_ms)
                                if _dinfo:
                                    disc_info = _dinfo

                        b_s, b_a, b_env_r, b_ns, b_d = buffer_r.sample(batch_size)
                        if running_ms:
                            b_s = running_ms.normalize(b_s)
                            b_ns = running_ms.normalize(b_ns)
                        b_disc_a = b_a
                        if use_gail_r and getattr(agent, "base_actor", None) is not None:
                            # BeTAIL：D 与代理奖励用执行动作 a=â+αã，不是残差 ã。
                            b_disc_a = agent.executed_from_residual(b_s, b_a)
                        if use_gail_r:
                            try:
                                b_gail_r = disc.predict_rewards(b_s, b_disc_a, to_numpy=False)
                            except TypeError:
                                b_gail_r = disc.predict_rewards(b_s, b_disc_a)
                            if torch.is_tensor(b_gail_r):
                                b_gail_r = b_gail_r.reshape(-1, 1)
                            else:
                                b_gail_r = np.asarray(b_gail_r, dtype=np.float32).reshape(-1, 1)

                            if gail_r_rms is not None:
                                b_gail_r = _gail_reward_normalize(
                                    b_gail_r, gail_r_rms, is_update=(gi == 0),
                                    z_clip=gail_reward_z_clip, std_floor=gail_reward_std_floor)

                            if torch.is_tensor(b_gail_r):
                                b_env_r = torch.as_tensor(
                                    b_env_r, dtype=torch.float32, device=b_gail_r.device
                                ).reshape(-1, 1)
                            else:
                                b_env_r = np.asarray(b_env_r, dtype=np.float32).reshape(-1, 1)
                            b_r = float(gail_reward_coef) * b_gail_r + env_reward_weight * b_env_r
                        else:
                            b_r = env_reward_weight * np.asarray(b_env_r, dtype=np.float32).reshape(-1, 1)

                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns,
                                           'rewards': b_r, 'dones': b_d}
                        if bc_coef > 0.0 and buffer_e is not None and buffer_e.size() > 0:
                            e_s, e_a = buffer_e.sample(
                                batch_size,
                                suffix_frac=expert_suffix_frac,
                                suffix_ratio=expert_suffix_ratio,
                            )
                            if running_ms:
                                e_s = running_ms.normalize(e_s)
                            transition_dict["expert_states"] = e_s
                            transition_dict["expert_actions"] = e_a
                            transition_dict["bc_coef"] = bc_coef
                        if bc_onpol_coef > 0.0:
                            transition_dict["bc_onpol_coef"] = bc_onpol_coef
                        try:
                            agent_info = agent.update(transition_dict, log_info=log_info)
                        except TypeError:
                            agent_info = agent.update(transition_dict)

                        if log_info:
                            avg_info.update(agent_info)
                            for k, v in disc_info.items():
                                avg_info[f"disc_{k}"] = v
                            # disc_expert_value：专家；disc_policy_value：当前 SAC 批次（近似 ρ_π）。
                            # buffer_g 上的 D 改记 disc_g_value（含 Trick1 预填，不是当前策略）。
                            # 残差时 D 打在执行动作 â+αã（b_disc_a），不是回放里的 ã。
                            if use_gail_r:
                                try:
                                    d_on = disc.predict_policy_prob(b_s, b_disc_a, to_numpy=True)
                                    if "disc_policy_value" in avg_info:
                                        avg_info["disc_g_value"] = avg_info["disc_policy_value"]
                                    avg_info["disc_policy_value"] = float(np.mean(np.asarray(d_on)))
                                    logged_d_pol = avg_info["disc_policy_value"]
                                except (TypeError, AttributeError):
                                    pass
                    agent.lr_decay(global_step)
                    if use_gail_r:
                        disc.lr_decay(global_step)

                    if 'disc_loss' in avg_info:
                        gail_loss.append(avg_info['disc_loss'])

                    if is_draw:
                        current_time = time.time()
                        delta_time = current_time - last_record_time
                        delta_steps = global_step - last_record_step
                        if delta_time > 0:
                            writer.add_scalar("charts/SPS", int(delta_steps / delta_time), global_step)
                        last_record_time = current_time
                        last_record_step = global_step
                        for k, v in avg_info.items():
                            writer.add_scalar(f"losses/{k}", v, global_step)

                    # 下一段采集对应 SB3 下一次 collect_rollouts 开头。
                    if hasattr(agent, "reset_noise"):
                        agent.reset_noise(batch_size=n_envs)

                global_step += n_envs
                pbar.update(n_envs)

                # ---------- 2.6 周期评估 ----------
                if (eval_env is not None
                        and global_step > learning_starts
                        and global_step % eval_interval == 0):
                    stats = rollout_eval(eval_env, agent, n_episodes=eval_episodes, running_ms=running_ms)
                    best_success = _handle_eval_checkpoint(
                        stats, global_step, agent, disc, running_ms, save_model_dir, is_save_model,
                        best_success)
                    if is_draw:
                        writer.add_scalar("eval/success_rate", stats['success_rate'], global_step)
                        writer.add_scalar("eval/return_mean", stats['return_mean'], global_step)
                        writer.add_scalar("eval/peak_force_mean", stats['peak_force_mean'], global_step)
                    pbar.write(
                        f"[eval@{global_step}] success={stats['success_rate']:.1%} "
                        f"ret={stats['return_mean']:.1f}±{stats['return_std']:.1f} "
                        f"peakF={stats['peak_force_mean']:.1f}N states={stats['state_dist']}")

    # ==================== 3. 训练结束：保存与终检 ====================
    if is_save_model and save_model_dir:
        agent.save_model(save_model_dir / "final_model")
        disc.save_model(save_model_dir / "final_model")
        if running_ms:
            running_ms.save_normalizer(save_model_dir / "final_model")
        final_info = {
            'best_success': float(best_success),
            'return': float(return_list[-1]) if return_list else None,
            'global_step': int(global_step),
            'episode': int(episodic_count),
        }
        (save_model_dir / "final_model").mkdir(parents=True, exist_ok=True)
        with open(save_model_dir / "final_model" / "info.json", 'w', encoding='utf-8') as f:
            json.dump(final_info, f, indent=2, ensure_ascii=False)

    restored_best = False
    if is_save_model and save_model_dir:
        restored_best = _load_gail_ckpt(agent, disc, running_ms, save_model_dir)

    if eval_env is not None:
        final_stats = rollout_eval(eval_env, agent, n_episodes=max(30, eval_episodes),
                                   running_ms=running_ms, seed_offset=200000)
        ckpt_tag = "best_success_model" if restored_best else "last_weights"
        print(f"[final eval @ {ckpt_tag}] success={final_stats['success_rate']:.1%} "
              f"ret={final_stats['return_mean']:.1f}±{final_stats['return_std']:.1f} "
              f"peakF={final_stats['peak_force_mean']:.1f}N")
        eval_env.close()

    env.close()
    writer.close()
    return np.asarray(return_list, dtype=np.float32), np.asarray(gail_loss, dtype=np.float32)
