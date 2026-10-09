
import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict
from pathlib import Path
from diffusers.optimization import get_scheduler
from training.model.gail.sac import PolicyNet, QValueNet
from training.policy.base_policy import BasePolicy
from training.common.checkpoint import load_state_dict, save_policy_cfg, save_state_dict

class SACPolicy(BasePolicy):
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_space,
                 optimizer: Dict, lr_scheduler: Dict, num_training_steps: int,
                 tau, gamma,
                 alpha=1.0,
                 log_std_init=-3,
                 autotune=True,
                 use_sde=True,
                 clip_mean=2.0,
                 alpha_min: float|None = None,
                 target_entropy: float|None = None,
                 actor_grad_clip_norm=1.0,
                 critic_grad_clip_norm=5.0,
                 absorbing_state=False):
        """
        optimizer: 各网络优化器超参（actor / critic / alpha），含 lr、betas、eps、weight_decay
        lr_scheduler: 学习率调度器配置（name、num_warmup_steps）
        num_training_steps: 总梯度更新次数，供调度器计算学习率曲线
        tau: 软更新因子, tau 通常设得非常小（如 0.005 或 0.001）。这意味着目标网络每步只向当前网络"挪动" 0.5% 或 0.1%，变化极其平滑。这能极大防止 Q 值过高估计和训练震荡
        gamma: 折扣因子
        use_sde: 是否使用 gSDE (generalized State-Dependent Exploration)
        log_std_init: log_std 初始值（gSDE 模式下）
        clip_mean: 限制 actor 均值输出范围
        target_entropy: 可覆盖默认的 -action_dim；
        """
        super().__init__()
        self.use_sde = use_sde
        self.action_space = action_space
        self.state_dim = int(state_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = int(action_dim)
        self.absorbing_state = bool(absorbing_state)
        self.physical_obs_dim = self.state_dim - int(self.absorbing_state)
        self.clip_mean = float(clip_mean)
        self.log_std_init = float(log_std_init)
        self.actor_grad_clip_norm = float(actor_grad_clip_norm)
        self.critic_grad_clip_norm = float(critic_grad_clip_norm)
        for name in ("actor_grad_clip_norm", "critic_grad_clip_norm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim,
                                         action_space, log_std_init=log_std_init,
                                         use_sde=use_sde,
                                         clip_mean=clip_mean).to(self.device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=optimizer['actor']['lr'],
            betas=optimizer['actor']['betas'],
            eps=optimizer['actor']['eps'],
            weight_decay=optimizer['actor']['weight_decay'],
        )
        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=optimizer['critic']['lr'],
            betas=optimizer['critic']['betas'],
            eps=optimizer['critic']['eps'],
            weight_decay=optimizer['critic']['weight_decay'],
        )
        self.actor_lr_scheduler = get_scheduler(
            lr_scheduler['name'],
            optimizer=self.actor_optimizer,
            num_warmup_steps=lr_scheduler['num_warmup_steps'],
            num_training_steps=num_training_steps,
        )
        self.critic_lr_scheduler = get_scheduler(
            lr_scheduler['name'],
            optimizer=self.critic_optimizer,
            num_warmup_steps=lr_scheduler['num_warmup_steps'],
            num_training_steps=num_training_steps,
        )
        self.gamma = gamma
        self.tau = tau
        self.autotune = autotune
        self.alpha_min = alpha_min  # alpha下限，防止温度坍缩到0导致gSDE探索消失
        self.bc_actor = None  # 冻结 BC，供 ρ_π 上的均值钉住

        self.alpha = alpha if torch.is_tensor(alpha) else torch.as_tensor(alpha, dtype=torch.float32, device=self.device )

        if autotune:
            self.target_entropy = -action_dim if target_entropy is None else float(target_entropy)
            self.log_alpha = self.alpha.detach().log().requires_grad_(True)
            self.alpha = self.log_alpha.exp().item()
            self.log_alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=optimizer['alpha']['lr'],
                betas=optimizer['alpha']['betas'],
                eps=optimizer['alpha']['eps'],
                weight_decay=optimizer['alpha']['weight_decay'],
            )
            self.alpha_lr_scheduler = get_scheduler(
                lr_scheduler['name'],
                optimizer=self.log_alpha_optimizer,
                num_warmup_steps=lr_scheduler['num_warmup_steps'],
                num_training_steps=num_training_steps,
            )
            

    def reset_noise(self, batch_size=1):
        """SB3：采集段开头 ``reset_noise(n_envs)``；``SAC.train`` 里 ``reset_noise()``。"""
        if self.use_sde:
            self.actor.reset_noise(batch_size)

    def predict_action(self, state, deterministic=False):
        """预测策略动作。"""
        states = np.asarray(state, dtype=np.float32)
        return_single = states.ndim == 1
        if return_single:
            states = states[None, :]
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            executed, _ = self.actor(states, deterministic=deterministic)
            if self.absorbing_state:
                executed = torch.where(
                    (states[..., self.physical_obs_dim] > 0.5)[:, None],
                    torch.zeros_like(executed), executed,
                )
        self.actor.train()
        executed_np = np.array(executed.detach().cpu().numpy(), dtype=np.float32, copy=True)
        if return_single:
            executed_np = executed_np[0]
        return executed_np

    def _absorbing_mask(self, states):
        return states[..., self.physical_obs_dim] > 0.5

    @torch.no_grad()
    def _calc_target(self, rewards, next_states, dones, alpha):
        next_actions, next_log_prob = self.actor(next_states)
        if self.absorbing_state:
            absorbing = self._absorbing_mask(next_states)[:, None]
            next_actions = torch.where(absorbing, torch.zeros_like(next_actions), next_actions)
            next_log_prob = torch.where(absorbing, torch.zeros_like(next_log_prob), next_log_prob)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_q_values = (torch.min(q1_value, q2_value) - alpha * next_log_prob).flatten()
        # DAC 将真正终止接入吸收态；吸收自环也持续 bootstrap。
        if not self.absorbing_state:
            next_q_values = (1 - dones.flatten()) * next_q_values
        return rewards.flatten() + self.gamma * next_q_values

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    @torch.no_grad()
    def update_targets(self):
        """由 trainer 决定调用时机，执行一次 Target Critic 软更新。"""
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def update(self, transition_dict, log_info=True):
        info = {}

        def _to_dev(x, extra_view=False):
            t = torch.as_tensor(x, dtype=torch.float32)
            if t.device != self.device:
                t = t.to(self.device)
            return t.view(-1, 1) if extra_view else t

        states = _to_dev(transition_dict['states'])
        actions = _to_dev(transition_dict['actions'])
        rewards = _to_dev(transition_dict['rewards'], extra_view=True)
        next_states = _to_dev(transition_dict['next_states'])
        dones = _to_dev(transition_dict['dones'], extra_view=True)

        # SB3 SAC.train：每个梯度步 reset_noise()（默认 batch=1，整 minibatch 共用一份 E）。
        # 采集用的 n_envs 份 E 由 train 循环在下一段采集开头重新抽。
        if self.use_sde:
            self.actor.reset_noise()

        obs = states
        next_obs = next_states

        # 计算 actions_pi 和 log_prob
        actions_pi, log_prob = self.actor(obs)
        log_prob = log_prob.reshape(-1, 1)

        active = ~self._absorbing_mask(states) if self.absorbing_state else torch.ones(
            states.shape[0], dtype=torch.bool, device=self.device,
        )
        has_active = bool(active.any())

        # 更新 alpha（计算用 tensor，避免每步 .item() 同步）
        if self.autotune:
            alpha_t = self.log_alpha.exp().detach()
            if has_active:
                alpha_loss = -(self.log_alpha * (log_prob[active] + self.target_entropy).detach()).mean()
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                self.alpha_lr_scheduler.step()
                if self.alpha_min is not None:
                    with torch.no_grad():
                        self.log_alpha.clamp_(min=float(np.log(self.alpha_min)))
            else:
                alpha_loss = torch.zeros((), device=self.device)
        else:
            alpha_t = self.alpha

        # 计算 target Q（使用 pre-step alpha）
        td_target = self._calc_target(rewards, next_obs, dones, alpha_t)

        # 计算 critic loss
        critic_1_values = self.critic_1(obs, actions).view(-1)
        critic_2_values = self.critic_2(obs, actions).view(-1)
        critic_1_loss = F.mse_loss(critic_1_values, td_target)
        critic_2_loss = F.mse_loss(critic_2_values, td_target)
        critic_loss = 0.5 * (critic_1_loss + critic_2_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        gn_critic = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.critic_grad_clip_norm,
        )
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()

        q1_pi = self.critic_1(obs, actions_pi)
        q2_pi = self.critic_2(obs, actions_pi)
        if has_active:
            actor_loss = (alpha_t * log_prob[active] - torch.min(q1_pi, q2_pi)[active]).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            gn_actor = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_grad_clip_norm,
            )
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()
        else:
            actor_loss = torch.zeros((), device=self.device)
            gn_actor = torch.zeros((), device=self.device)

        if log_info:
            if self.autotune:
                self.alpha = self.log_alpha.exp().item()
            info = {
                "critic_value": (critic_1_values.mean().item() + critic_2_values.mean().item()) / 2.0,
                "critic_loss": critic_loss.item(),
                "alpha": self.alpha if isinstance(self.alpha, float) else float(self.alpha),
                "gn_critic": gn_critic.item(),
                "gn_actor": gn_actor.item(),
            }
            info["actor_loss"] = actor_loss.item()
            if self.autotune:
                info["alpha_loss"] = alpha_loss.item()
            info["actor_loss"] = actor_loss.item()
        return info
    
    def load_policy(self, model_dir):
        _model_dir = Path(model_dir)
        self.actor.load_state_dict(load_state_dict(_model_dir, "policy_net", map_location=self.device))

    def load_model(self, model_dir):
        _model_dir = Path(model_dir)
        self.load_policy(_model_dir)
        self.critic_1.load_state_dict(load_state_dict(_model_dir, "qvalue_net1", map_location=self.device))
        self.critic_2.load_state_dict(load_state_dict(_model_dir, "qvalue_net2", map_location=self.device))
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def save_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        save_state_dict(self.actor.state_dict(), _model_dir, "policy_net")
        save_state_dict(self.critic_1.state_dict(), _model_dir, "qvalue_net1")
        save_state_dict(self.critic_2.state_dict(), _model_dir, "qvalue_net2")
        cfg = self.actor.export_cfg()
        cfg.update(absorbing_state=self.absorbing_state, physical_obs_dim=self.physical_obs_dim)
        save_policy_cfg(cfg, _model_dir)


class ResidualSACPolicy(SACPolicy):
    """以当前观测和基座动作为条件，输出缩放前的残差动作。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_dim = self.state_dim - self.action_dim
        self.physical_obs_dim = self.obs_dim - int(self.absorbing_state)
