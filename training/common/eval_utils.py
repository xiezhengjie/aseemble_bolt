import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import AutoresetMode


def _value_at(value, index):
    """读取 vector info 中指定环境槽位的值。"""
    if isinstance(value, dict):
        return value
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return array.item()
    return array[index]


def _info_at(infos, index):
    """把 Gymnasium 的 dict-of-arrays info 转成单环境字典。"""
    if not isinstance(infos, dict):
        return {}
    result = {}
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        if not bool(_value_at(infos.get("_" + key, True), index)):
            continue
        result[key] = (_info_at(value, index) if isinstance(value, dict)
                       else _value_at(value, index))
    return result


def _terminal_infos(infos, index):
    """读取 SAME_STEP 下指定槽位的终止 info 及其 EpisodeStats 内层 info。"""
    outer = _info_at(infos.get("final_info", {}), index)
    inner = outer.get("final_info", {})
    if not isinstance(inner, dict):
        inner = {}
    return outer, inner


class Evaluator:
    """评估器基类，支持单环境和 Gymnasium VectorEnv。"""
    def __init__(self, agent):
        self.agent = agent

    def _episode(self, env, seed):
        """评估单个episode。"""
        ep_peak_force = 0.0
        done = False
        obs, _ = env.reset(seed=seed)
        while not done:
            action = self.agent.predict_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            force = np.asarray(info.get("force", ()), dtype=np.float32).reshape(-1)
            ep_peak_force = max(ep_peak_force, float(np.linalg.norm(force[:6])))
        return ep_peak_force, info

    @staticmethod
    def _stats(records):
        peak_forces = [record["peak_force"] for record in records]
        returns = [record["return"] for record in records]
        lengths = [record["length"] for record in records]
        successes = [record["success"] for record in records]
        final_prev_states = [record["prev_state"] for record in records
                             if record["prev_state"] is not None]
        prev_states, counts = np.unique(final_prev_states, return_counts=True)
        return {
            "success_rate": float(np.mean(successes)),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "length_mean": float(np.mean(lengths)),
            "peak_force_mean": float(np.mean(peak_forces)),
            "state_dist": {int(state): int(count)
                           for state, count in zip(prev_states, counts)},
            "n_episodes": len(records),
        }

    @staticmethod
    def _record(ordinary_info, terminal_outer, terminal_inner, peak_force):
        """合并终止 info，保留与单环境评估相同的优先级。"""
        return {
            "peak_force": float(peak_force),
            "return": float(terminal_inner.get(
                "episodic_return", terminal_outer.get("episodic_return", 0.0))),
            "length": int(terminal_inner.get(
                "episodic_length", terminal_outer.get("episodic_length", 0))),
            "success": float(terminal_outer.get(
                "success", terminal_inner.get("success", ordinary_info.get("success", False)))),
            "prev_state": terminal_outer.get("prev_state"),
        }

    def _vector_actions(self, obs, active, env):
        actions = np.zeros(
            (env.num_envs,) + env.single_action_space.shape,
            dtype=env.single_action_space.dtype,
        )
        indices = np.flatnonzero(active)
        if indices.size:
            predicted = np.asarray(
                self.agent.predict_action(obs[indices], deterministic=True),
                dtype=env.single_action_space.dtype,
            )
            if predicted.ndim == 1:
                predicted = predicted[None, :]
            actions[indices] = predicted
        return actions

    def _vector_reset(self, obs, env):
        return None

    def _vector_step(self, state, obs, next_obs, active, ended):
        return None

    def _evaluate_vector(self, env, n_episodes, seed_offset):
        if getattr(env, "autoreset_mode", AutoresetMode.SAME_STEP) != AutoresetMode.SAME_STEP:
            raise ValueError("并行评估环境必须使用 autoreset_mode=AutoresetMode.SAME_STEP")

        records = []
        n_envs = int(env.num_envs)
        for batch_start in range(0, n_episodes, n_envs):
            batch_size = min(n_envs, n_episodes - batch_start)
            if seed_offset is None:
                seeds = None
            else:
                seeds = [int(seed_offset) + batch_start + index for index in range(n_envs)]
            obs, _ = env.reset(seed=seeds)
            obs = np.asarray(obs, dtype=np.float32)
            active = np.zeros(n_envs, dtype=bool)
            active[:batch_size] = True
            peaks = np.zeros(n_envs, dtype=np.float32)
            state = self._vector_reset(obs, env)

            while np.any(active):
                active_before = active.copy()
                actions = self._vector_actions(obs, active, env)
                next_obs, _, terminated, truncated, infos = env.step(actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                ended = np.asarray(terminated) | np.asarray(truncated)
                self._vector_step(state, obs, next_obs, active_before, ended)

                for index in np.flatnonzero(active_before):
                    info = _info_at(infos, index)
                    if ended[index]:
                        terminal_outer, terminal_inner = _terminal_infos(infos, index)
                        force = terminal_outer.get("force", ())
                    else:
                        force = info.get("force", ())
                    force = np.asarray(force, dtype=np.float32).reshape(-1)
                    if force.size:
                        peaks[index] = max(
                            peaks[index], float(np.linalg.norm(force[:6]))
                        )

                    if ended[index]:
                        records.append(self._record(
                            terminal_outer, terminal_outer, terminal_inner, peaks[index],
                        ))
                        active[index] = False
                obs = next_obs

        return self._stats(records)

    def evaluate(self, env, n_episodes=20, seed_offset=100000):
        """评估环境；seed_offset=None 时延续环境已有的随机数序列。"""
        n_episodes = int(n_episodes)
        if n_episodes <= 0:
            raise ValueError("n_episodes 必须为正数")
        if isinstance(env, gym.vector.VectorEnv):
            return self._evaluate_vector(env, n_episodes, seed_offset)

        records = []
        for ep in range(n_episodes):
            seed = None if seed_offset is None else int(seed_offset) + ep
            ep_peak_force, info = self._episode(env, seed)
            episode_info = info.get("final_info", {})
            records.append(self._record(
                info, info, episode_info, ep_peak_force,
            ))
        return self._stats(records)


class BaseChunkPolicyEvaluator(Evaluator):
    """基础块策略评估器。"""
    def __init__(self, agent, obs_horizon, action_interval, sampling_steps):
        super().__init__(agent)
        self.obs_horizon = int(obs_horizon)
        self.action_interval = int(action_interval)
        self.sampling_steps = int(sampling_steps)

    def _episode(self, env, seed):
        """评估单个episode。"""
        ep_peak_force = 0.0
        done = False
        obs, _ = env.reset(seed=seed)
        history = [np.asarray(obs, dtype=np.float32)] * self.obs_horizon
        self.agent.reset()
        action_plan = None
        plan_index = 0
        while not done:
            if action_plan is None or plan_index >= min(self.action_interval, len(action_plan)):
                obs_window = np.stack(history)
                obs_tensor = torch.from_numpy(obs_window).unsqueeze(0)
                action_plan = self.agent.sample(obs_tensor, self.sampling_steps)[0].cpu().numpy()
                plan_index = 0
            action = action_plan[plan_index]
            plan_index += 1
            obs, _, terminated, truncated, info = env.step(action)
            history = (history + [np.asarray(obs, dtype=np.float32)])[-self.obs_horizon:]
            done = bool(terminated or truncated)
            force = np.asarray(info["force"], dtype=np.float32).reshape(-1)
            ep_peak_force = max(ep_peak_force, float(np.linalg.norm(force[:6])))

        return ep_peak_force, info

    def _vector_reset(self, obs, env):
        self.agent.reset()
        self._vector_state = {
            "history": np.repeat(obs[:, None, :], self.obs_horizon, axis=1),
            "plans": None,
            "plan_indices": np.full(env.num_envs, self.action_interval, dtype=np.int64),
            "prev_naction": None,
        }
        return self._vector_state

    @torch.no_grad()
    def _sample_vector_plans(self, state, indices):
        tensor = torch.from_numpy(state["history"][indices])
        device = getattr(self.agent, "device", None)
        if device is not None:
            tensor = tensor.to(device)
        previous = getattr(self.agent, "prev_naction", None)
        try:
            if hasattr(self.agent, "prev_naction"):
                self.agent.prev_naction = (
                    None if state["prev_naction"] is None
                    else state["prev_naction"][indices].clone()
                )
            plans = self.agent.sample(tensor, self.sampling_steps)
            prior = getattr(self.agent, "prev_naction", None)
            if prior is not None:
                if state["prev_naction"] is None:
                    state["prev_naction"] = prior.new_zeros(
                        (state["history"].shape[0],) + tuple(prior.shape[1:])
                    )
                state["prev_naction"][indices] = prior
        finally:
            if hasattr(self.agent, "prev_naction"):
                self.agent.prev_naction = previous
        if torch.is_tensor(plans):
            plans = plans.detach().cpu().numpy()
        return np.asarray(plans, dtype=np.float32)

    def _vector_base_actions(self, state, active, env):
        actions = np.zeros(
            (env.num_envs,) + env.single_action_space.shape,
            dtype=env.single_action_space.dtype,
        )
        indices = np.flatnonzero(active)
        if not indices.size:
            return actions
        if state["plans"] is None:
            replan = indices
        else:
            plan_length = min(self.action_interval, state["plans"].shape[1])
            replan = indices[state["plan_indices"][indices] >= plan_length]
        if replan.size:
            plans = self._sample_vector_plans(state, replan)
            if state["plans"] is None:
                state["plans"] = np.empty(
                    (env.num_envs,) + plans.shape[1:], dtype=np.float32,
                )
            state["plans"][replan] = plans
            state["plan_indices"][replan] = 0
        actions[indices] = state["plans"][indices, state["plan_indices"][indices]]
        return actions

    def _vector_actions(self, obs, active, env):
        return self._vector_base_actions(self._vector_state, active, env)

    def _vector_step(self, state, obs, next_obs, active, ended):
        state["history"] = np.concatenate(
            [state["history"][:, 1:], next_obs[:, None, :]], axis=1,
        )
        state["plan_indices"] += active.astype(np.int64)
        ended_active = active & ended
        state["history"][ended_active] = next_obs[ended_active, None, :]
        state["plan_indices"][ended_active] = self.action_interval
        if state["prev_naction"] is not None:
            state["prev_naction"][ended_active] = 0


class ResidualEvaluator(BaseChunkPolicyEvaluator):
    """残差评估器。"""
    def __init__(self, base_agent, residual_agent, residual_scale,
                 obs_horizon, action_interval, sampling_steps):
        super().__init__(base_agent, obs_horizon, action_interval, sampling_steps)
        self.residual_agent = residual_agent
        self.residual_scale = float(residual_scale)

    def _episode(self, env, seed):
        """评估单个episode。"""
        ep_peak_force = 0.0
        done = False
        obs, _ = env.reset(seed=seed)
        history = [np.asarray(obs, dtype=np.float32)] * self.obs_horizon
        self.agent.reset()
        action_plan = None
        plan_index = 0
        while not done:
            if action_plan is None or plan_index >= min(self.action_interval, len(action_plan)):
                obs_window = np.stack(history)
                obs_tensor = torch.from_numpy(obs_window).unsqueeze(0)
                action_plan = self.agent.sample(obs_tensor, self.sampling_steps)[0].cpu().numpy()
                plan_index = 0
            base_action = action_plan[plan_index]
            residual_obs = np.concatenate([obs, base_action], axis=-1)
            residual_action = self.residual_agent.predict_action(residual_obs, deterministic=True)
            action = np.clip(base_action + self.residual_scale * residual_action,
                             env.action_space.low, env.action_space.high)
            plan_index += 1
            obs, _, terminated, truncated, info = env.step(action)
            history = (history + [np.asarray(obs, dtype=np.float32)])[-self.obs_horizon:]
            done = bool(terminated or truncated)
            force = np.asarray(info["force"], dtype=np.float32).reshape(-1)
            ep_peak_force = max(ep_peak_force, float(np.linalg.norm(force[:6])))

        return ep_peak_force, info

    def _vector_actions(self, obs, active, env):
        base_actions = self._vector_base_actions(self._vector_state, active, env)
        actions = base_actions.copy()
        indices = np.flatnonzero(active)
        if indices.size:
            residual_obs = np.concatenate([obs[indices], base_actions[indices]], axis=-1)
            residual = np.asarray(
                self.residual_agent.predict_action(residual_obs, deterministic=True),
                dtype=actions.dtype,
            )
            if residual.ndim == 1:
                residual = residual[None, :]
            actions[indices] = np.clip(
                base_actions[indices] + self.residual_scale * residual,
                env.single_action_space.low,
                env.single_action_space.high,
            )
        return actions
