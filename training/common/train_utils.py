import json
import math
import time
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import deque
from gymnasium.vector import AutoresetMode
from training.common.eval_utils import Evaluator, ResidualEvaluator
from training.common.rl_utils import RewardScaling

_EPISODE_INFO_KEYS = ("depth", "position_error_xy", "yaw_error", "angle_z")

# ---------------------------------------------------------------------------
# 数据采集
# ---------------------------------------------------------------------------
class DataCollector:
    """跨更新步维护观测与回合统计；replay 只存环境原始 obs / reward。

    子类只需覆盖 `_act`，返回 (actions, extra)：
      - actions 会传给 env.step 与 add_batch
      - extra 会展开进 add_batch（如 base_actions / res_actions）
    """

    def __init__(self, env, agent, replay_buffer):
        if env.autoreset_mode != AutoresetMode.SAME_STEP:
            raise ValueError("训练环境必须使用 autoreset_mode=AutoresetMode.SAME_STEP")
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.n_envs = env.num_envs
        self.episode_returns = np.zeros(self.n_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.n_envs, dtype=np.int64)
        self._last_ended = np.zeros(self.n_envs, dtype=bool)

    def reset(self, seed, options=None):
        """训练开始时调用一次；之后回合由 vector env 自动重置。"""
        obs, _ = self.env.reset(seed=seed, options=options)
        self.obs = np.asarray(obs, dtype=np.float32).copy()
        self.env.action_space.seed(seed)
        self.episode_returns.fill(0)
        self.episode_lengths.fill(0)
        self._last_ended.fill(False)

    def _action(self, warmup: bool) -> tuple[np.ndarray, dict]:
        if warmup:
            return self.env.action_space.sample(), {}
        self.agent.reset_noise(self.n_envs)
        return self.agent.predict_action(self.obs), {}

    def step(self, warmup: bool = False) -> list[dict]:
        actions, extra = self._action(warmup)
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        ended = terminated | truncated

        # SAME_STEP 下 next_obs 是重置后的首帧，TD target 必须使用 final_obs。
        td_next_obs = next_obs.copy()
        successes = np.zeros(self.n_envs, dtype=np.float32)
        episodes: list[dict] = []
        self.episode_returns += rewards
        self.episode_lengths += 1

        for index in np.flatnonzero(ended):
            final_info = infos["final_info"]
            td_next_obs[index] = infos["final_obs"][index]
            success = bool(final_info["success"][index]) if "success" in final_info else False
            successes[index] = success
            record = {
                "episodic_return": float(self.episode_returns[index]),
                "episodic_length": int(self.episode_lengths[index]),
                "success": success,
            }
            for key in _EPISODE_INFO_KEYS:
                if key in final_info:
                    record[key] = float(final_info[key][index])
            episodes.append(record)

        self.replay_buffer.add_batch(
            obs=self.obs, actions=actions, next_obs=td_next_obs,
            rewards=rewards, dones=terminated, successes=successes,
            **extra,
        )
        self.episode_returns[ended] = 0
        self.episode_lengths[ended] = 0
        self.obs = next_obs.copy()
        self._last_ended = ended
        return episodes


class ResidualDataCollector(DataCollector):
    """基座按块规划，残差逐步执行；history / plan_indices 在下一次 _act 里 lazy 更新。"""

    def __init__(
        self, env, base_agent, res_agent, replay_buffer,
        residual_scale, obs_horizon, action_interval, sampling_steps,
    ):
        super().__init__(env, res_agent, replay_buffer)
        self.base_agent = base_agent
        self.residual_scale = float(residual_scale)
        self.obs_horizon = int(obs_horizon)
        self.action_interval = int(action_interval)
        self.sampling_steps = int(sampling_steps)
        if (not math.isfinite(self.residual_scale) or self.residual_scale < 0
                or self.obs_horizon < 1 or self.action_interval < 1
                or self.sampling_steps < 1):
            raise ValueError(
                "residual_scale 必须非负有限，窗口、动作间隔和采样步数必须为正数"
            )

    def reset(self, seed, options=None):
        super().reset(seed, options)
        self.history = np.repeat(self.obs[:, None, :], self.obs_horizon, axis=1)
        self.action_plans = None
        self.plan_indices = np.zeros(self.n_envs, dtype=np.int64)
        self._pending_history = False

    def _action(self, warmup: bool) -> tuple[np.ndarray, dict]:
        # lazy 更新：把上一次 step 后的 obs 追加进 history，并处理回合结束。
        if self._pending_history:
            self.history = np.concatenate(
                [self.history[:, 1:], self.obs[:, None, :]], axis=1,
            )
            self.history[self._last_ended] = self.obs[self._last_ended, None, :]
            self.plan_indices[self._last_ended] = self.action_interval
        self._pending_history = True

        base_actions = self._plan_base_actions()
        res_actions = np.zeros_like(base_actions)
        if not warmup:
            self.agent.reset_noise(self.n_envs)
            res_actions = np.asarray(self.agent.predict_action(self.obs), dtype=np.float32)

        actions = np.clip(
            base_actions + self.residual_scale * res_actions,
            self.env.single_action_space.low,
            self.env.single_action_space.high,
        )
        # 保存缩放前的残差；环境与判别器使用裁剪后的执行动作。
        return actions, {"base_actions": base_actions, "res_actions": res_actions}

    @torch.no_grad()
    def _plan_base_actions(self) -> np.ndarray:
        if self.action_plans is None:
            indices = np.arange(self.n_envs)
        else:
            # 块耗尽时重新规划，允许 action_interval 大于基座输出长度。
            plan_length = min(self.action_interval, self.action_plans.shape[1])
            indices = np.flatnonzero(self.plan_indices >= plan_length)

        if indices.size:
            obs_tensor = torch.from_numpy(self.history[indices])
            plans = self.base_agent.sample(obs_tensor.to(self.base_agent.device), self.sampling_steps)
            plans = plans.detach().cpu().numpy().astype(np.float32, copy=False)
            if (plans.ndim != 3 or plans.shape[0] != len(indices)
                    or plans.shape[1] < 1
                    or plans.shape[2:] != self.env.single_action_space.shape):
                raise ValueError("基座 sample 必须返回 (num_envs, action_horizon, action_dim)")

            if self.action_plans is None:
                self.action_plans = np.empty(
                    (self.n_envs,) + plans.shape[1:], dtype=np.float32,
                )
            self.action_plans[indices] = plans
            self.plan_indices[indices] = 0

        actions = self.action_plans[np.arange(self.n_envs), self.plan_indices].copy()
        self.plan_indices += 1
        return actions


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SupervisedPolicyTrainer:
    """行为克隆训练器，支持残差策略。"""

    def __init__(
        self, agent, train_loader, val_loader, total_epochs,
        model_dir, patience=100, min_delta=1e-6,
        wb_run=None, is_early_stop=True,
    ):
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_epochs = int(total_epochs)
        self.model_dir = model_dir
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.is_early_stop = bool(is_early_stop)
        self.wb_run = wb_run

    def train(self):
        """独立窗口随机组 batch，按验证 MSE 保存最佳模型并早停。"""
        best_val = float("inf")
        patience_count = 0
        try:
            with tqdm(total=int(self.total_epochs)) as progress:
                for epoch in range(int(self.total_epochs)):
                    train_loss = self.agent.fit_epoch(self.train_loader)
                    val_loss = self.agent.eval_epoch(self.val_loader)
                    
                    if hasattr(self.agent, "lr_decay"):
                        self.agent.lr_decay(epoch)
                    if self.wb_run is not None:
                        self.wb_run.log({"loss/train": train_loss, "loss/val": val_loss}, step=epoch)
                    if val_loss < best_val - self.min_delta:
                        best_val = val_loss
                        patience_count = 0
                        self.agent.save_model(self.model_dir)
                    else:
                        patience_count += 1
                    progress.set_postfix(
                        epoch=epoch + 1, train_loss=f"{train_loss:.4f}",
                        val_loss=f"{val_loss:.4f}", best_val=f"{best_val:.4f}",
                        patience=f"{patience_count}/{self.patience}")
                    progress.update(1)
                    if self.patience > 0 and patience_count >= self.patience and self.is_early_stop:
                        break
        finally:
            pass
        return best_val


class OffPolicyTrainer:
    """统一采集、预热、多次策略更新、日志、评估和保存。

    - eval_env 必须是独立单环境
    - replay.sample 返回 SAC batch 字典
    - learning_starts 按 transition 数计（包含预热）
    - gradient_step 累计策略梯度更新数，控制 Target 更新间隔
    - 学习率按 global_step / total_timesteps 线性衰减
    """

    collector_class = DataCollector

    def __init__(
        self, env, eval_env, agent, replay_buffer, total_timesteps,
        learning_starts=1000, batch_size=256, *, collector=None, evaluator=None,
        seed=1, eval_interval=5000, eval_episodes=10, final_eval_episodes=30,
        save_model_dir="models", is_save_model=True, 
        is_draw=True, log_interval=100, wb_run=None,
        policy_warmup=False, reset_options=None, policy_updates=1,
        target_update_interval=1, policy_lr_min_ratio=0.3,
    ):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.total_timesteps = int(total_timesteps)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.policy_updates = int(policy_updates)
        self.target_update_interval = int(target_update_interval)
        self.policy_lr_min_ratio = policy_lr_min_ratio
        self.seed = seed
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.final_eval_episodes = int(final_eval_episodes)
        self.log_interval = int(log_interval)
        self.is_save_model = is_save_model
        self.save_model_dir = Path(save_model_dir)
        self.policy_warmup = policy_warmup
        self.reset_options = reset_options

        self.collector = collector or self.collector_class(env, agent, replay_buffer)
        self.evaluator = evaluator or Evaluator(agent)

        self.global_step = 0
        self.gradient_step = 0
        self.return_list: list[float] = []
        self.success_history: deque[bool] = deque(maxlen=50)
        self.best_success = -float("inf")
        self.final_stats: dict = {}
        self.wb_run = wb_run if is_draw else None
        self._pending_logs = {}
        self._last_log_step = -1

    def _log(self, values: dict) -> None:
        self._pending_logs.update(values)

    def _flush_logs(self) -> None:
        if self.wb_run is not None and self._pending_logs:
            step = max(self.global_step, self._last_log_step + 1)
            self.wb_run.log(self._pending_logs, step=step)
            self._last_log_step = step
        self._pending_logs.clear()

    def train(self) -> np.ndarray:
        """总步数向上取整到 num_envs；环境由调用方负责关闭。"""
        next_eval, next_log = self.eval_interval, self.log_interval
        last_time, last_step = time.perf_counter(), 0
        try:
            self.collector.reset(self.seed, options=self.reset_options)
            with tqdm(total=self.total_timesteps, desc=type(self).__name__, mininterval=0.5) as bar:
                while self.global_step < self.total_timesteps:
                    warmup = self.global_step < self.learning_starts and not self.policy_warmup 
                    episodes = self.collector.step(warmup=warmup)
                    self.global_step += self.env.num_envs

                    should_log = self.global_step >= next_log
                    info = (
                        self._update(should_log)
                        if self.global_step > self.learning_starts else {}
                    )

                    for episode in episodes:
                        self.return_list.append(episode["episodic_return"])
                        self.success_history.append(episode["success"])
                    for key in episodes[0] if episodes else ():
                        self._log({
                            f"charts/{key}": np.mean([
                                episode[key] for episode in episodes if key in episode
                            ])
                        })
                    if episodes:
                        self._log({
                            "charts/avg_episodic_return": np.mean(self.return_list[-10:]),
                            "charts/success_rate_50ep": np.mean(self.success_history),
                        })
                    bar.update(self.env.num_envs)

                    if should_log:
                        now = time.perf_counter()
                        self._log({"charts/SPS": (self.global_step - last_step) / (now - last_time)})
                        self._log({f"losses/{key}": value for key, value in info.items()})
                        last_time, last_step = now, self.global_step
                        next_log = (
                            self.global_step // self.log_interval + 1
                        ) * self.log_interval

                    if (self.global_step >= next_eval and self.global_step > self.learning_starts):
                        self._evaluate(final=False)
                        next_eval = (self.global_step // self.eval_interval + 1) * self.eval_interval
                    self._flush_logs()

            self.final_stats = self._evaluate(final=True)
            self._save_checkpoint("final_model", self.final_stats)
        finally:
            self._flush_logs()
            if self.wb_run is not None:
                self.wb_run.finish()
        return np.asarray(self.return_list, dtype=np.float32)

    def _prepare_batch(self, batch: dict) -> dict:
        """保留回放字段，并为 SAC 提供 states / next_states。"""
        batch = dict(batch)
        batch["states"] = batch["obs"]
        batch["next_states"] = batch["next_obs"]
        return batch

    def _lr_scale(self, min_ratio: float) -> float:
        return max(min_ratio, 1.0 - self.global_step / self.total_timesteps)

    def _update(self, log_info: bool) -> dict:
        self.agent.set_lr_scale(self._lr_scale(self.policy_lr_min_ratio))
        info: dict = {}
        for index in range(self.policy_updates):
            batch = self._prepare_batch(self.replay_buffer.sample(self.batch_size))
            info = self.agent.update(
                batch, log_info=log_info and index == self.policy_updates - 1,
            )
            self.gradient_step += 1
            if self.gradient_step % self.target_update_interval == 0:
                self.agent.update_targets()
        return info

    def _evaluate(self, final: bool = False) -> dict:
        stats = self.evaluator.evaluate(
            self.eval_env,
            n_episodes=self.final_eval_episodes if final else self.eval_episodes,
            seed_offset=200000 if final else 100000,
        )
        prefix = "final_eval" if final else "eval"
        self._log({
            f"{prefix}/{key}": stats[key]
            for key in ("success_rate", "return_mean", "peak_force_mean")
        })

        if not final and stats["success_rate"] > self.best_success:
            self.best_success = stats["success_rate"]
            self._save_checkpoint("best_success_model", stats)

        tqdm.write(
            f"[{prefix}@{self.global_step}] "
            f"success={stats['success_rate']:.1%} "
            f"return={stats['return_mean']:.2f}±{stats['return_std']:.2f}"
        )
        return stats

    def _save_checkpoint(self, name: str, stats: dict) -> None:
        if not self.is_save_model:
            return
        path = self.save_model_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.agent.save_model(path)
        info = {
            "global_step": self.global_step,
            "episode": len(self.return_list),
            "return": self.return_list[-1] if self.return_list else None,
            "best_success": self.best_success if np.isfinite(self.best_success) else None,
            "evaluation": stats,
        }
        with (path / "info.json").open("w", encoding="utf-8") as file:
            json.dump(info, file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# GAIL
# ---------------------------------------------------------------------------
class GAILTrainer(OffPolicyTrainer):
    """在 SAC 更新前重算奖励，更新后连续训练判别器。"""

    def __init__(
        self, *args, discriminator, expert_buffer, generator_buffer,
        disc_updates=2, disc_batch_size=256, env_reward_weight=0.0,
        gail_reward_coef=1.0, gail_reward_scale=True, gamma=0.99,
        success_reward=100.0, disc_lr_min_ratio=0.1, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.expert_buffer = expert_buffer
        self.generator_buffer = generator_buffer
        self.disc_updates = int(disc_updates)
        self.disc_lr_min_ratio = float(disc_lr_min_ratio)
        self.disc_batch_size = int(disc_batch_size)
        self.env_reward_weight = float(env_reward_weight)
        self.gail_reward_coef = float(gail_reward_coef)
        self.reward_scale = RewardScaling((1,), gamma) if gail_reward_scale else None
        self.success_reward = float(success_reward)
        self.disc_losses: list[float] = []

    def _discriminator_states(self, states: np.ndarray) -> np.ndarray:
        return states

    def _prepare_batch(self, batch: dict) -> dict:
        batch = super()._prepare_batch(batch)
        reward = self.discriminator.predict_rewards(
            self._discriminator_states(batch["states"]),
            batch["actions"], to_numpy=True,
        ).reshape(-1, 1)
        if self.reward_scale is not None:
            reward = self.reward_scale(reward)

        batch["rewards"] = (
            self.gail_reward_coef * reward
            + self.env_reward_weight * batch["rewards"].reshape(-1, 1)
            + self.success_reward * batch["successes"].reshape(-1, 1)
        )
        return batch

    def _update(self, log_info: bool) -> dict:
        # 判别器在全部策略更新完成后才更新。
        info = super()._update(log_info)

        self.discriminator.set_lr_scale(self._lr_scale(self.disc_lr_min_ratio))
        disc_info: dict = {}
        for index in range(self.disc_updates):
            expert_batch = self.expert_buffer.sample(self.disc_batch_size)
            gen_states, gen_actions = self.generator_buffer.sample(self.disc_batch_size)
            disc_info = self.discriminator.update(
                self._discriminator_states(expert_batch["obs"]),
                expert_batch["actions"],
                self._discriminator_states(gen_states),
                gen_actions,
                log_info=log_info and index == self.disc_updates - 1,
            )

        info.update({f"disc_{key}": value for key, value in disc_info.items()})
        if log_info:
            self.disc_losses.append(disc_info["loss"])
        return info

    def _save_checkpoint(self, name: str, stats: dict) -> None:
        super()._save_checkpoint(name, stats)
        if self.is_save_model:
            self.discriminator.save_model(self.save_model_dir / name)


class ResidualGAILTrainer(GAILTrainer):
    """使用相同的基座分块参数采集和评估残差策略。"""

    collector_class = ResidualDataCollector

    def __init__(
        self, env, eval_env, base_agent, res_agent, replay_buffer,
        total_timesteps, learning_starts=1000, batch_size=256, *,
        residual_scale, obs_horizon, action_interval, sampling_steps,
        evaluator=None, **kwargs,
    ):

        collector = self.collector_class(
            env, base_agent, res_agent, replay_buffer, residual_scale,
            obs_horizon, action_interval, sampling_steps,
        )
        if evaluator is None:
            evaluator = ResidualEvaluator(
                base_agent, res_agent, residual_scale,
                obs_horizon, action_interval, sampling_steps,
            )

        super().__init__(
            env, eval_env, res_agent, replay_buffer, total_timesteps,
            learning_starts, batch_size,
            collector=collector, evaluator=evaluator, policy_warmup=True, **kwargs,
        )

    def _discriminator_states(self, states: np.ndarray) -> np.ndarray:
        states = super()._discriminator_states(states)
        # replay.obs 尾部包含基座动作；专家池与 GenWindowView 已经是原始观测。
        return states[..., : self.env.single_observation_space.shape[-1]]

    def _prepare_batch(self, batch: dict) -> dict:
        batch = super()._prepare_batch(batch)
        raw_obs_dim = int(self.env.single_observation_space.shape[-1])
        batch["states"] = batch["obs"][..., :raw_obs_dim]
        batch["next_states"] = batch["next_obs"]
        batch["actions"] = batch["res_actions"]
        return batch
