from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from training.policy.base_policy import BasePolicy
from training.model.base.conditional_unet1d import ConditionalUnet1D
from training.common.checkpoint import load_state_dict, save_state_dict


class DiffusionPolicy(BasePolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        warmstart_timestep=50,
        eta=0.0,
        lr=1e-4,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.inference_noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        self.warmstart_timestep = int(warmstart_timestep)
        if not 0 <= self.warmstart_timestep < noise_scheduler.config.num_train_timesteps:
            raise ValueError("warmstart_timestep 必须位于训练扩散 timestep 范围内")
        self.eta = float(eta)
        self.register_buffer('prev_naction', None, persistent=False)

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs
        if min(horizon, n_obs_steps, n_action_steps) < 1 or n_action_steps > horizon:
            raise ValueError("horizon 必须覆盖观测偏移和整个 action chunk")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def reset(self):
        """新 episode 开始前清除上一回合的动作先验。"""
        self.prev_naction = None

    # ── 采样 ────────────────────────────────────────────────────────
    @torch.no_grad()
    def conditional_sample(
        self, condition_data, condition_mask,
        global_cond=None, generator=None, **kwargs,
    ):
        """按参考实现，用上轮移位动作加噪后执行 DDIM 去噪。

        每次调用对应执行 n_action_steps 步后的重新规划；首次使用零动作先验。
        保留参考实现的完整 inference timesteps 调度。
        """
        model = self.model
        scheduler = self.inference_noise_scheduler

        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        if (self.prev_naction is None or self.prev_naction.shape != condition_data.shape
                or self.prev_naction.device != condition_data.device
                or self.prev_naction.dtype != condition_data.dtype):
            self.prev_naction = torch.zeros_like(condition_data)
        scheduler.set_timesteps(self.num_inference_steps, device=condition_data.device)
        timesteps = torch.full((condition_data.shape[0],), self.warmstart_timestep,
                               device=condition_data.device, dtype=torch.long)
        trajectory = scheduler.add_noise(self.prev_naction, noise, timesteps)

        for t in scheduler.timesteps:
            # 每步开始前把条件位置写回，防止上一步 scheduler 更新污染
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                trajectory, t,
                global_cond=global_cond,
            )
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, eta=self.eta, **kwargs,
            ).prev_sample

        # 结束时再写回一次
        trajectory[condition_mask] = condition_data[condition_mask]
        # 与参考实现一致：左移已执行的步数，未覆盖的尾部保持原先验值。
        remaining = self.horizon - self.n_action_steps
        self.prev_naction[:, :remaining] = trajectory[:, self.n_action_steps:].detach()
        return trajectory

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """从观测字典预测动作序列。obs_dict 必须包含 'obs'。"""
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict, "past_action 作为条件尚未实现"

        obs = obs_dict["obs"].to(device=self.device, dtype=self.dtype)
        batch_size, obs_steps, obs_dim = obs.shape

        global_cond = obs[:, :self.n_obs_steps].reshape(batch_size, -1)
        cond_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        action_pred = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            **self.kwargs,
        )

        action = action_pred[:, :self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred,
            'action_full': action_pred,
        }

        return result

    @torch.no_grad()
    def sample(self, obs, sampling_steps=None):
        """供 chunk evaluator 使用的张量接口。"""
        old_steps = self.num_inference_steps
        try:
            if sampling_steps is not None:
                self.num_inference_steps = int(sampling_steps)
            return self.predict_action({'obs': obs})['action']
        finally:
            self.num_inference_steps = old_steps

    # ── 训练损失 ────────────────────────────────────────────────────
    def compute_loss(self, batch):
        assert 'valid_mask' not in batch, "训练阶段不应出现 valid_mask"

        obs = batch['obs']          # (B, To, obs_dim)
        action = batch['action']    # (B, Ta, action_dim)

        global_cond = None
        trajectory = action

        global_cond = obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)

        # 前向扩散
        noise = torch.randn_like(trajectory)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        pred = self.model(
            noisy_trajectory, timesteps,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 只在未知位置算 loss
        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()

    # ── 优化一步 ────────────────────────────────────────────────────
    def update(self, batch: dict) -> torch.Tensor:
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        self.model.train()
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    # ── 一轮训练 / 验证 ─────────────────────────────────────────────
    @staticmethod
    def _batches(loader_or_obs, action, batch_size, shuffle=False, drop_last=False):
        if action is None:
            yield from loader_or_obs
        else:
            # 保留旧的直接传入 obs/action 张量接口。
            from training.common.buffer_utils import TensorBatchLoader
            for obs_batch, action_batch in TensorBatchLoader(
                    loader_or_obs, action, batch_size, shuffle, drop_last):
                yield {'obs': obs_batch, 'action': action_batch}

    def fit_epoch(self, loader_or_obs, action=None, batch_size=256, drop_last=False) -> float:
        total = torch.zeros((), device=self.device)
        count = 0
        for batch in self._batches(loader_or_obs, action, batch_size, True, drop_last):
            loss = self.update(batch)
            n = len(batch['obs'])
            total = total + loss * n
            count += n
        if count == 0:
            raise ValueError("训练 DataLoader 没有可用样本")
        return float(total / count)

    @torch.no_grad()
    def eval_epoch(self, loader_or_obs, action=None, batch_size=1024) -> float:
        """验证仍会重新采样噪声和 timestep，loss 有随机性。"""
        self.model.eval()
        total = torch.zeros((), device=self.device)
        count = 0
        for batch in self._batches(loader_or_obs, action, batch_size):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.compute_loss(batch)
            n = len(batch['obs'])
            total = total + loss * n
            count += n
        if count == 0:
            raise ValueError("验证 DataLoader 没有可用样本")
        return float(total / count)

    # ── 存取 ────────────────────────────────────────────────────────
    def load_model(self, model_dir):
        self.model.load_state_dict(load_state_dict(
            Path(model_dir), "policy_net", map_location=self.device,
        ))
        self.reset()

    def save_model(self, model_dir):
        save_state_dict(self.model.state_dict(), model_dir, "policy_net")
