import json
import random
import math
import time
import torch
import numpy as np
import wandb

from tqdm import tqdm
from pathlib import Path
from collections import deque
from gymnasium.vector import AutoresetMode
from training.common.eval_utils import Evaluator, ResidualEvaluator
from training.common.rl_utils import RewardScaling, set_seed

_EPISODE_INFO_KEYS = ("depth", "position_error_xy", "yaw_error", "angle_z")

# ---------------------------------------------------------------------------
# 数据采集
# ---------------------------------------------------------------------------
class DataCollector:
    """跨更新步维护观测；回合统计读取 EpisodeStatsWrapper 的 final_info。

    子类覆盖 `_action`，返回 (actions, extra)：
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

    def reset(self, seed, options=None):
        """训练开始时调用一次；之后回合由 vector env 自动重置。"""
        obs, _ = self.env.reset(seed=seed, options=options)
        self.obs = np.asarray(obs, dtype=np.float32).copy()
        self.env.action_space.seed(seed)

    def _action(self, warmup: bool) -> tuple[np.ndarray, dict]:
        if warmup:
            return self.env.action_space.sample(), {}
        self.agent.reset_noise(self.n_envs)
        return self.agent.predict_action(self.obs), {}

    def _transition_extra(self, next_obs, td_next_obs, terminated, ended, extra):
        return extra

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
        final_info = infos.get("final_info", {})
        final_info = final_info.get("final_info", final_info)

        for index in np.flatnonzero(ended):
            td_next_obs[index] = infos["final_obs"][index]
            record = {"success": False}
            for key in ("episodic_return", "episodic_length", "success", *_EPISODE_INFO_KEYS):
                if key not in final_info:
                    continue
                value = np.asarray(final_info[key], dtype=object)
                value = value.item() if value.ndim == 0 else value[index]
                if key == "episodic_length":
                    record[key] = int(value)
                elif key == "success":
                    record[key] = bool(value)
                    successes[index] = float(record[key])
                else:
                    record[key] = float(value)
            episodes.append(record)

        extra = self._transition_extra(next_obs, td_next_obs, terminated, ended, extra)
        self.replay_buffer.add_batch(
            obs=self.obs, actions=actions, next_obs=td_next_obs,
            rewards=rewards, dones=terminated, successes=successes,
            **extra,
        )
        self.obs = next_obs.copy()
        return episodes


class ResidualDataCollector(DataCollector):
    """基座按块规划，下一基座动作同时用于 TD target 和下一步执行。"""

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
        self.prev_naction = None
        self.base_actions = self._plan_base_actions()

    def _action(self, warmup: bool) -> tuple[np.ndarray, dict]:
        base_actions = self.base_actions.copy()
        res_actions = np.zeros_like(base_actions)
        if not warmup:
            self.agent.reset_noise(self.n_envs)
            residual_obs = np.concatenate([self.obs, base_actions], axis=-1)
            res_actions = np.asarray(self.agent.predict_action(residual_obs), dtype=np.float32)

        actions = np.clip(
            base_actions + self.residual_scale * res_actions,
            self.env.single_action_space.low,
            self.env.single_action_space.high,
        )
        # 保存缩放前的残差；环境与判别器使用裁剪后的执行动作。
        return actions, {"base_actions": base_actions, "res_actions": res_actions}

    def _transition_extra(self, next_obs, td_next_obs, terminated, ended, extra):
        self.history = np.concatenate(
            [self.history[:, 1:], td_next_obs[:, None, :]], axis=1,
        )
        self.plan_indices += 1

        next_base_actions = np.zeros_like(self.base_actions)
        indices = np.flatnonzero(~terminated)
        if indices.size:
            next_base_actions[indices] = self._plan_base_actions(indices)

        # timeout 仍从旧回合终帧 bootstrap；实际执行从新回合重新规划。
        self.history[ended] = next_obs[ended, None, :]
        self.plan_indices[ended] = self.action_interval
        if self.prev_naction is not None:
            self.prev_naction[ended] = 0
        self.base_actions = self._plan_base_actions()
        return {**extra, "next_base_actions": next_base_actions}

    @torch.no_grad()
    def _sample_base_plans(self, indices):
        obs_tensor = torch.from_numpy(self.history[indices]).to(self.base_agent.device)
        previous = self.base_agent.prev_naction
        try:
            # diffusion 先验按环境保存，子批次重规划和评估不能串用历史。
            self.base_agent.prev_naction = (
                None if self.prev_naction is None else self.prev_naction[indices].clone()
            )
            plans = self.base_agent.sample(obs_tensor, self.sampling_steps)
            prior = self.base_agent.prev_naction
            if prior is not None:
                if self.prev_naction is None:
                    self.prev_naction = prior.new_zeros((self.n_envs,) + prior.shape[1:])
                self.prev_naction[indices] = prior
        finally:
            self.base_agent.prev_naction = previous
        return plans.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def _plan_base_actions(self, indices=None) -> np.ndarray:
        if indices is None:
            indices = np.arange(self.n_envs)
        if self.action_plans is None:
            replan = indices
        else:
            # 块耗尽时重新规划，允许 action_interval 大于基座输出长度。
            plan_length = min(self.action_interval, self.action_plans.shape[1])
            replan = indices[self.plan_indices[indices] >= plan_length]

        if replan.size:
            plans = self._sample_base_plans(replan)
            if self.action_plans is None:
                self.action_plans = np.empty(
                    (self.n_envs,) + plans.shape[1:], dtype=np.float32,
                )
            self.action_plans[replan] = plans
            self.plan_indices[replan] = 0

        return self.action_plans[indices, self.plan_indices[indices]].copy()


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
            with tqdm(total=int(self.total_epochs), dynamic_ncols=True, ascii=True) as progress:
                for epoch in range(int(self.total_epochs)):
                    train_loss = self.agent.fit_epoch(self.train_loader)
                    val_loss = self.agent.eval_epoch(self.val_loader)
                    
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

    - eval_env 必须独立于训练环境；支持单环境或 SAME_STEP VectorEnv
    - replay.sample 返回 SAC batch 字典
    - learning_starts 按 transition 数计（包含预热）
    - use_warmup=True 时前 learning_starts 步采集预热动作（随机 / 纯基座），期间不做更新
    - gradient_step 累计策略梯度更新数，控制 Target 更新间隔
    - 学习率由各策略内部的 lr_scheduler 按 num_training_steps 调度
    """

    collector_class = DataCollector

    def __init__(
        self, env, eval_env, agent, replay_buffer, total_timesteps,
        learning_starts=1000, batch_size=256, *, collector=None, evaluator=None,
        seed=1, eval_interval=5000, eval_episodes=10, final_eval_episodes=30,
        save_model_dir="models", is_save_model=True, 
        is_draw=True, log_interval=100, wb_run=None,
        use_warmup=True, reset_options=None, policy_updates=1,
        target_update_interval=1, eval_seed=None, eval_env_seed=None,
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
        self.seed = seed
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.final_eval_episodes = int(final_eval_episodes)
        self.eval_seed = None if eval_seed is None else int(eval_seed)
        self.eval_env_seed = None if eval_env_seed is None else int(eval_env_seed)
        self.log_interval = int(log_interval)
        self.is_save_model = is_save_model
        self.save_model_dir = Path(save_model_dir)
        self.use_warmup = use_warmup
        self.reset_options = reset_options

        self.collector = collector or self.collector_class(env, agent, replay_buffer)
        self.evaluator = evaluator or Evaluator(agent)

        self.global_step = 0
        self.gradient_step = 0
        self.return_list: list[float] = []
        self.ep_length_list: deque[int] = deque(maxlen=30)
        self.success_history: deque[bool] = deque(maxlen=30)
        self.episode_info_history = {
            key: deque(maxlen=30) for key in _EPISODE_INFO_KEYS
        }
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
        window_elapsed, last_step = 0.0, self.global_step
        try:
            self.collector.reset(self.seed, options=self.reset_options)
            with tqdm(total=self.total_timesteps, desc=type(self).__name__, dynamic_ncols=True, ascii=True, mininterval=0.5) as bar:
                while self.global_step < self.total_timesteps:
                    step_started = time.perf_counter()
                    warmup = self.global_step < self.learning_starts and self.use_warmup
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
                        self.ep_length_list.append(episode["episodic_length"])
                        for key in _EPISODE_INFO_KEYS:
                            if key in episode:
                                self.episode_info_history[key].append(episode[key])

                    if episodes:
                        for key in _EPISODE_INFO_KEYS:
                            history = self.episode_info_history[key]
                            if history:
                                self._log({f"tasks/{key}": np.mean(history)})

                        self._log({
                            "charts/episodic_return_30ep": np.mean(self.return_list[-30:]),
                            "charts/success_rate_30ep": np.mean(self.success_history),
                            "charts/episode_length_30ep": np.mean(self.ep_length_list),
                        })

                    bar.update(self.env.num_envs)
                    # Evaluation and log flush must not enter the training throughput window.
                    window_elapsed += time.perf_counter() - step_started

                    if should_log:
                        self._log({"charts/SPS": (self.global_step - last_step) / window_elapsed})
                        self._log({f"losses/{key}": value for key, value in info.items()})
                        window_elapsed, last_step = 0.0, self.global_step
                        next_log = (
                            self.global_step // self.log_interval + 1
                        ) * self.log_interval

                    if (self.global_step >= next_eval and self.global_step > self.learning_starts):
                        self._evaluate()
                        next_eval = (self.global_step // self.eval_interval + 1) * self.eval_interval
                    self._flush_logs()

            self._save_checkpoint("final_model", {})
        finally:
            self._flush_logs()
        return np.asarray(self.return_list, dtype=np.float32)

    def _prepare_batch(self, batch: dict) -> dict:
        """保留回放字段，并为 SAC 提供 states / next_states。"""
        batch = dict(batch)
        batch["states"] = batch["obs"]
        batch["next_states"] = batch["next_obs"]
        return batch

    def _update(self, log_info: bool) -> dict:
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

    def _evaluate(self) -> dict:
        rng_state = None
        if self.eval_seed is not None:
            rng_state = (
                random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            )
            set_seed(self.eval_seed)
        try:
            stats = self.evaluator.evaluate(
                self.eval_env,
                n_episodes=self.eval_episodes,
                seed_offset=self.eval_env_seed,
            )
        finally:
            if rng_state is not None:
                python_state, numpy_state, torch_state, cuda_states = rng_state
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
        self._log({
            f"eval/{key}": stats[key]
            for key in ("success_rate", "return_mean", "peak_force_mean")
        })

        if stats["success_rate"] > self.best_success:
            self.best_success = stats["success_rate"]
            self._save_checkpoint("best_success_model", stats)

        tqdm.write(
            f"[eval@{self.global_step}] "
            f"success={stats['success_rate']:.1%} "
            f"return={stats['return_mean']:.2f}±{stats['return_std']:.2f}"
        )
        return stats

    def record_final_evaluation(self, stats: dict) -> None:
        """记录训练入口执行的终验结果并更新最终模型信息。"""
        self.final_stats = stats
        self._log({
            f"final_eval/{key}": stats[key]
            for key in ("success_rate", "return_mean", "peak_force_mean")
        })
        self._save_checkpoint("final_model", stats)
        self._flush_logs()
        tqdm.write(
            f"[final_eval@{self.global_step}] "
            f"success={stats['success_rate']:.1%} "
            f"return={stats['return_mean']:.2f}±{stats['return_std']:.2f}"
        )

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
        self._upload_checkpoint(path)

    def _upload_checkpoint(self, path: Path) -> None:
        """把已写入的 checkpoint 文件同步到当前 W&B run。"""
        if self.wb_run is None:
            return
        for model_path in sorted(path.iterdir()):
            if model_path.is_file():
                wandb.save(
                    str(model_path.resolve()),
                    base_path=str(path.parent.resolve()), policy="now",
                )


# ---------------------------------------------------------------------------
# GAIL
# ---------------------------------------------------------------------------
class GAILTrainer(OffPolicyTrainer):
    """在 SAC 更新前重算奖励，更新后连续训练判别器。"""

    def __init__(
        self, *args, discriminator, expert_buffer, generator_buffer,
        disc_updates=2, disc_batch_size=256, env_reward_weight=0.0,
        gail_reward_coef=1.0, gail_reward_scale=True, gamma=0.99,
        success_reward=100.0, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.expert_buffer = expert_buffer
        self.generator_buffer = generator_buffer
        self.disc_updates = int(disc_updates)
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
        if not self.is_save_model:
            return
        path = self.save_model_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.discriminator.save_model(path)
        super()._save_checkpoint(name, stats)


class ResidualGAILTrainer(GAILTrainer):
    """使用相同的基座分块参数采集和评估残差策略。"""

    collector_class = ResidualDataCollector

    def __init__(
        self, env, eval_env, base_agent, res_agent, replay_buffer,
        total_timesteps, learning_starts=1000, batch_size=256, *,
        residual_scale, obs_horizon, action_interval, sampling_steps,
        use_warmup=False, evaluator=None, **kwargs,
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
            collector=collector, evaluator=evaluator,
            use_warmup=use_warmup, **kwargs,
        )

    def _discriminator_states(self, states: np.ndarray) -> np.ndarray:
        states = super()._discriminator_states(states)
        # replay.obs 尾部包含基座动作；专家池与 GenWindowView 已经是原始观测。
        return states[..., : self.env.single_observation_space.shape[-1]]

    def _prepare_batch(self, batch: dict) -> dict:
        batch = super()._prepare_batch(batch)
        batch["actions"] = batch["res_actions"]
        return batch
