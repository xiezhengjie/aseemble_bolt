
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from training.model.gail.sac import PolicyNet, QValueNet
from training.policy.base_policy import BasePolicy
from training.common.checkpoint import load_state_dict, save_policy_cfg, save_state_dict

class SACPolicy(BasePolicy):
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_space,
                 actor_lr, critic_lr, alpha_lr, tau, gamma,
                 alpha=1.0,
                 log_std_init=-3,
                 autotune=True,
                 use_sde=True,
                 use_orthogonal_init=False,
                 clip_mean=2.0,
                 alpha_min: float|None = None,
                 target_entropy: float|None = None):
        """
        actor_lr: Actor网络学习率, 值太大，策略会剧烈变化，导致训练震荡甚至发散；如果太小，策略收敛极慢，可能陷入局部最优
        critic_lr: Critic网络学习率, 设置 critic_lr 略大于或等于 actor_lr
        alpha_lr: 温度学习率
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
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.clip_mean = float(clip_mean)
        self.log_std_init = float(log_std_init)
        self.use_orthogonal_init = bool(use_orthogonal_init)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim,
                                         action_space, log_std_init=log_std_init,
                                         use_sde=use_sde, use_orthogonal_init=use_orthogonal_init,
                                         clip_mean=clip_mean).to(self.device)
        self.critic_1 = QValueNet(state_dim, hidden_dim,
                                            action_dim, use_orthogonal_init).to(self.device)
        self.critic_2 = QValueNet(state_dim, hidden_dim,
                                            action_dim, use_orthogonal_init).to(self.device)
        self.target_critic_1 = QValueNet(state_dim,
                                                   hidden_dim, action_dim, use_orthogonal_init).to(self.device)
        self.target_critic_2 = QValueNet(state_dim,
                                                   hidden_dim, action_dim, use_orthogonal_init).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr, eps=1e-5)
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
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, eps=1e-5)
            

    def reset_noise(self, batch_size=1):
        """SB3：采集段开头 ``reset_noise(n_envs)``；``SAC.train`` 里 ``reset_noise()``。"""
        if self.use_sde:
            self.actor.reset_noise(batch_size)

    def _obs_to_batch(self, state):
        """整理观测为网络输入，返回 (tensor[B,...], return_single)。"""
        states = np.asarray(state, dtype=np.float32)
        if states.ndim == 1:
            states = states[np.newaxis, :]
            return_single = True
        else:
            return_single = False
        t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return t, return_single

    def _residual_obs(self, states):
        return states

    def predict_action(self, state, deterministic=False, base_only=False):
        """取普通 SAC 的环境动作。``base_only`` 仅由残差子类实现。"""
        if base_only:
            raise ValueError("普通 SACAgent 不支持 base_only")
        states, return_single = self._obs_to_batch(state)
        states = self.normalize_obs(states)
        self.actor.eval()
        with torch.no_grad():
            executed, _ = self.actor(states, deterministic=deterministic)
        self.actor.train()
        executed_np = np.array(executed.detach().cpu().numpy(), dtype=np.float32, copy=True)
        if return_single:
            executed_np = executed_np[0]
        return executed_np

    def calc_target(self, rewards, next_states, dones):
        """ 计算目标Q值 """
        next_states = self.normalize_obs(next_states)
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_states)
            q1_value = self.target_critic_1(next_states, next_actions)
            q2_value = self.target_critic_2(next_states, next_actions)
            next_q_values = torch.min(q1_value, q2_value) - self.alpha * next_log_prob
        td_target = rewards.flatten() + self.gamma * (1 - dones.flatten()) * next_q_values.view(-1)
        return td_target

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

        states = self.normalize_obs(_to_dev(transition_dict['states']))
        actions = _to_dev(transition_dict['actions'])
        rewards = _to_dev(transition_dict['rewards'], extra_view=True)
        next_states = self.normalize_obs(_to_dev(transition_dict['next_states']))
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

        # 更新 alpha（计算用 tensor，避免每步 .item() 同步）
        if self.autotune:
            # 先取出 pre-step α，给 target Q / actor；再更新 log_alpha。
            alpha_t = self.log_alpha.exp().detach()
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            if self.alpha_min is not None:
                with torch.no_grad():
                    self.log_alpha.clamp_(min=float(np.log(self.alpha_min)))
        else:
            alpha_t = self.alpha 

        # 计算 target Q（使用 pre-step alpha）
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_obs)
            q1_value = self.target_critic_1(next_obs, next_actions)
            q2_value = self.target_critic_2(next_obs, next_actions)
            next_q_values = torch.min(q1_value, q2_value) - alpha_t * next_log_prob
        td_target = rewards.flatten() + self.gamma * (1 - dones.flatten()) * next_q_values.view(-1)

        # 计算 critic loss
        critic_1_values = self.critic_1(obs, actions).view(-1)
        critic_2_values = self.critic_2(obs, actions).view(-1)
        critic_1_loss = F.mse_loss(critic_1_values, td_target)
        critic_2_loss = F.mse_loss(critic_2_values, td_target)
        critic_loss = 0.5 * (critic_1_loss + critic_2_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 5.0)
        self.critic_optimizer.step()

        q1_pi = self.critic_1(obs, actions_pi)
        q2_pi = self.critic_2(obs, actions_pi)
        actor_loss = (alpha_t * log_prob - torch.min(q1_pi, q2_pi)).mean()
      
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        if log_info:
            if self.autotune:
                self.alpha = self.log_alpha.exp().item()
            info = {
                "critic_1_value": critic_1_values.mean().item(),
                "critic_2_value": critic_2_values.mean().item(),
                "critic_loss": critic_loss.item(),
                "alpha": self.alpha if isinstance(self.alpha, float) else float(self.alpha),
            }
            if self.autotune:
                info["alpha_loss"] = alpha_loss.item()
            info["actor_loss"] = actor_loss.item()
        return info
    
    def set_lr_scale(self, scale):
        """按初始学习率设置倍率；衰减进度由 trainer 计算。"""
        for p in self.actor_optimizer.param_groups:
            p['lr'] = self.actor_lr * scale
        for p in self.critic_optimizer.param_groups:
            p['lr'] = self.critic_lr * scale
        if self.autotune:
            for p in self.log_alpha_optimizer.param_groups:
                p['lr'] = self.alpha_lr * scale

    def load_policy(self, model_dir):
        _model_dir = Path(model_dir)
        self.actor.load_state_dict(load_state_dict(_model_dir, "policy_net", map_location=self.device))
        self._load_obs_normalizer(_model_dir)

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
        self._save_obs_normalizer(_model_dir)
        save_state_dict(self.critic_1.state_dict(), _model_dir, "qvalue_net1")
        save_state_dict(self.critic_2.state_dict(), _model_dir, "qvalue_net2")
        cfg = self.actor.export_cfg()
        save_policy_cfg(cfg, _model_dir)
