import torch
import numpy as np

class  Evaluator:
    """评估器基类。"""
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

    def evaluate(self, env, n_episodes=20, seed_offset=100000):
        """评估环境；seed_offset=None 时延续环境已有的随机数序列。"""
        peak_forces, final_prev_states, returns, lengths, successes = [], [], [], [], []
    
        for ep in range(n_episodes):
            seed = None if seed_offset is None else seed_offset + ep
            ep_peak_force, info = self._episode(env, seed)
            episode_info = info.get("final_info", {})
            returns.append(float(episode_info.get("episodic_return", 0.0)))
            lengths.append(int(episode_info.get("episodic_length", 0)))
            successes.append(float(info.get("success", episode_info.get("success", False))))
            if "prev_state" in info:
                final_prev_states.append(info["prev_state"])
            peak_forces.append(ep_peak_force)

        prev_states, counts = np.unique(final_prev_states, return_counts=True)
        return {
            'success_rate':    float(np.mean(successes)),
            'return_mean':     float(np.mean(returns)),
            'return_std':      float(np.std(returns)),
            'length_mean':     float(np.mean(lengths)),
            'peak_force_mean': float(np.mean(peak_forces)),
            'state_dist':      {int(s): int(c) for s, c in zip(prev_states, counts)},
            'n_episodes':      int(n_episodes),
        }


class BaseChunkPolicyEvaluator(Evaluator):
    """基础块策略评估器。"""
    def __init__(self, agent, obs_horizon, action_interval, sampling_steps):
        super().__init__(agent)
        self.obs_horizon = obs_horizon
        self.action_interval = action_interval
        self.sampling_steps = sampling_steps


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


class ResidualEvaluator(BaseChunkPolicyEvaluator):
    """残差评估器。"""
    def __init__(self, base_agent, residual_agent, residual_scale, 
                 obs_horizon, action_interval, sampling_steps):
        super().__init__(base_agent, obs_horizon, action_interval, sampling_steps)
        self.action_interval = action_interval
        self.sampling_steps = sampling_steps
        self.residual_agent = residual_agent
        self.residual_scale = residual_scale


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




