import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from utils.checkpoint import has_weight, load_policy_cfg, load_state_dict, save_policy_cfg, save_state_dict
from utils.rl_utils import orthogonal_init

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def current_frame(states, seq_len, raw_obs_dim):
    """(B, T, D) / 展平叠帧 / 单帧 → 当前帧 (B, D)。

    供残差模块与判别器使用：基座编码器需要整段历史，
    残差 actor/critic 与 D_φ 按 Markovian 只看当前帧（定稿 §2）。
    """
    x = states if torch.is_tensor(states) else torch.as_tensor(
        states, dtype=torch.float32)
    t = None if seq_len is None else int(seq_len)
    d = int(raw_obs_dim)
    if x.dim() == 3:
        return x[:, -1, :]
    if x.dim() == 2:
        if t is not None and t > 1:
            if x.shape[-1] == t * d:
                return x.view(x.shape[0], t, d)[:, -1, :]
            raise ValueError(
                f"展平叠帧期望最后一维 {t * d}，实际 {x.shape[-1]}"
                f"（seq_len={t}, raw_obs_dim={d}）")
        return x
    if x.dim() == 1:
        if t is not None and t > 1 and x.numel() == t * d:
            return x.view(t, d)[-1].unsqueeze(0)
        return x.unsqueeze(0)
    raise ValueError(
        f"current_frame 不支持 shape={tuple(x.shape)}"
        f"（seq_len={t}, raw_obs_dim={d}）")


class GRUObsEncoder(nn.Module):
    """时序矩阵 (T, D) → LayerNorm(h)。

    LayerNorm 随 GRU 在 BC 阶段联合训练、进入 RL 阶段一并冻结；
    残差策略与 Q 用归一化表示 h_norm（三模块同一表示，铁律 2）。
    seq_len=None 时不使用本类。
    """

    def __init__(self, seq_len, raw_obs_dim, gru_hidden_dim, use_h_norm=False):
        super().__init__()
        self.seq_len = int(seq_len)
        self.raw_obs_dim = int(raw_obs_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru = nn.GRU(self.raw_obs_dim, self.gru_hidden_dim, batch_first=True)
        self.use_h_norm = bool(use_h_norm)
        if self.use_h_norm:
            self.h_norm = nn.LayerNorm(self.gru_hidden_dim)  # 带仿射；铁律 2
        else:
            self.h_norm = None

    @property
    def norm_dim(self):
        """归一化表示 h_norm 的维度。"""
        return self.gru_hidden_dim

    @property
    def joint_dim(self):
        """拼接输入 [s; h_norm] 的维度（Lin 2025 式，见定稿 §5）。"""
        return self.norm_dim + self.raw_obs_dim

    def hidden_only(self, x, normalized=True):
        """(T, D) → h 或 h_norm（normalized=True），不带当前帧拼接。"""
        seq = self.as_sequence(x)
        _, hn = self.gru(seq)
        h = hn[-1]
        if normalized and self.h_norm is not None:
            h = self.h_norm(h)
        return h

    def as_sequence(self, x):
        """整理为 (B, T, D)。兼容 (T, D) / (B, T, D) / 展平 (B, T*D)。"""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            if x.shape[-1] == self.raw_obs_dim and x.shape[-2] == self.seq_len:
                return x.unsqueeze(0)
            if x.shape[-1] == self.seq_len * self.raw_obs_dim:
                return x.view(x.shape[0], self.seq_len, self.raw_obs_dim)
            raise ValueError(
                f"无法把 shape={tuple(x.shape)} 解释为 (T, D) 或 (B, T*D)；"
                f"期望 T={self.seq_len}, D={self.raw_obs_dim}")
        if x.dim() == 1:
            return x.view(1, self.seq_len, self.raw_obs_dim)
        raise ValueError(f"GRU 输入维度错误: dim={x.dim()}, shape={tuple(x.shape)}")

    def forward(self, x):
        seq = self.as_sequence(x)
        current = seq[:, -1, :]
        h = self.hidden_only(seq, normalized=True)
        return torch.cat([h, current], dim=1)


class PolicyNetContinuous(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, action_space,
                 log_std_init=-3, use_sde=True, use_orthogonal_init=False,
                 clip_mean=2.0, seq_len=None, raw_obs_dim=10, gru_hidden_dim=64,
                 cond_action_dim=0, use_h_norm=False):
        super(PolicyNetContinuous, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = list(hidden_dim)
        self.use_sde = use_sde
        self.log_std_init = log_std_init
        self.clip_mean = clip_mean
        self.cond_action_dim = int(cond_action_dim)

        self.encoder = None
        if seq_len is not None:
            self.encoder = GRUObsEncoder(seq_len, raw_obs_dim, gru_hidden_dim,
                                         use_h_norm=use_h_norm)

        layers=[]
        input_dim = self.encoder.joint_dim if self.encoder is not None else state_dim
        input_dim += self.cond_action_dim
        for output_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.fc_latent = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim[-1], action_dim)
        # 限制均值输出范围
        if clip_mean > 0:
            self.mu_clamp = nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean)
        else:
            self.mu_clamp = nn.Identity()

        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

        if use_sde:
            # gSDE: log_std 为全局参数，shape = [latent_sde_dim, action_dim]
            # SB3 Actor 默认 learn_features=False：φ 对噪声/方差 detach，梯度不进特征
            self.learn_features = False
            self._sde_eps = 1e-6
            self.log_std = nn.Parameter(
                torch.ones(hidden_dim[-1], action_dim) * log_std_init,
                requires_grad=True
            )
            self.exploration_mat = None
            self.exploration_matrices = None
            self.reset_noise()
        else:
            self.learn_features = False
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        if use_orthogonal_init:
            orthogonal_init(self.fc_latent)
            orthogonal_init(self.fc_mu, gain=0.01)

    def reset_noise(self, batch_size=1):
        """SB3 StateDependentNoiseDistribution.sample_weights。

        E ~ rsample(N(0, σ))，保留对 log_std 的重参数化梯度。
        同时采样 exploration_mat 与 exploration_matrices，与 SB3 一致。
        """
        if not self.use_sde:
            return
        std = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = weights_dist.rsample()
        self.exploration_matrices = weights_dist.rsample((batch_size,))

    def get_noise(self, latent_sde):
        """SB3 StateDependentNoiseDistribution.get_noise。

        采集 ``reset_noise(n_envs)`` 后用每环境一份 E；训练
        ``reset_noise()``（batch=1）后整批走 ``exploration_mat``。
        """
        if self.exploration_mat is None:
            self.reset_noise()
        if not self.learn_features:
            latent_sde = latent_sde.detach()
        exploration_mat = self.exploration_mat.to(device=latent_sde.device, dtype=latent_sde.dtype)
        exploration_matrices = self.exploration_matrices.to(
            device=latent_sde.device, dtype=latent_sde.dtype)
        n_batch = latent_sde.shape[0]
        if n_batch == 1 or n_batch != exploration_matrices.shape[0]:
            return torch.mm(latent_sde, exploration_mat)
        return torch.bmm(latent_sde.unsqueeze(1), exploration_matrices).squeeze(1)

    def _cat_cond(self, feat, cond):
        """BeTAIL：残差策略输入 [s, â]。"""
        if self.cond_action_dim <= 0:
            return feat
        if cond is None:
            raise ValueError("残差策略需要 cond=π_H(s)")
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=feat.dtype, device=feat.device)
        else:
            cond = cond.to(device=feat.device, dtype=feat.dtype)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if cond.shape[0] != feat.shape[0] or cond.shape[-1] != self.cond_action_dim:
            raise ValueError(
                f"cond shape={tuple(cond.shape)} 与 feat batch={feat.shape[0]} "
                f"cond_dim={self.cond_action_dim} 不匹配")
        return torch.cat([feat, cond], dim=1)

    def _latent(self, x, cond=None):
        if self.encoder is not None:
            x = self.encoder(x)
        return self.fc_latent(self._cat_cond(x, cond))

    def forward(self, x, deterministic=False, cond=None):
        latent = self._latent(x, cond=cond)
        mu = self.mu_clamp(self.fc_mu(latent))
        std = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
            
        if self.use_sde:
            # SB3 proba_distribution：learn_features=False 时对 φ detach 再算方差
            latent_sde = latent if self.learn_features else latent.detach()
            variance = torch.mm(latent_sde.pow(2), std.pow(2))
            action_std = torch.sqrt(variance + self._sde_eps)

            if deterministic:
                x_t = mu
            else:
                x_t = mu + self.get_noise(latent)

            normal = Normal(mu, action_std)
        else:
            normal = Normal(mu, std)
            if deterministic:
                x_t = mu
            else:
                x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def mean_action_from_hidden(self, s, h_norm, cond=None):
        """基座头路径：s 为当前帧、h_norm 为冻结 GRU 的归一化表示。

        输出即 a_base = BC(s, h)：跳过编码器（h 已在外部算好），
        只走 [s; h_norm] → latent → μ → tanh·scale。
        """
        if h_norm is None:
            raise ValueError("mean_action_from_hidden 需要 h_norm（冻结 GRU 的输出）")
        if not torch.is_tensor(h_norm):
            h_norm = torch.as_tensor(h_norm, dtype=torch.float32)
        if h_norm.dim() == 1:
            h_norm = h_norm.unsqueeze(0)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if s.shape[-1] != (self.encoder.raw_obs_dim if self.encoder is not None else s.shape[-1]):
            raise ValueError(
                f"当前帧维度不符: s.shape={tuple(s.shape)}")
        x = torch.cat([h_norm, s], dim=1)
        latent = self.fc_latent(self._cat_cond(x, cond))
        mu = self.mu_clamp(self.fc_mu(latent))
        return torch.tanh(mu) * self.action_scale + self.action_bias

    def mean_action(self, x, cond=None):
        """确定性均值动作（BC / eval），不走 gSDE 与 log_prob。"""
        latent = self._latent(x, cond=cond)
        mu = self.mu_clamp(self.fc_mu(latent))
        return torch.tanh(mu) * self.action_scale + self.action_bias

    def export_cfg(self) -> dict:
        enc = self.encoder
        scale = self.action_scale.detach().cpu().flatten()
        bias = self.action_bias.detach().cpu().flatten()
        return {
            "hidden_dim": list(self.hidden_dim),
            "action_dim": int(self.action_dim),
            "seq_len": None if enc is None else int(enc.seq_len),
            "raw_obs_dim": None if enc is None else int(enc.raw_obs_dim),
            "gru_hidden_dim": None if enc is None else int(enc.gru_hidden_dim),
            "use_h_norm": bool(enc.use_h_norm) if enc is not None else False,
            "clip_mean": float(self.clip_mean),
            "log_std_init": float(self.log_std_init),
            "use_sde": bool(self.use_sde),
            "cond_action_dim": int(self.cond_action_dim),
            "action_low": (bias - scale).tolist(),
            "action_high": (bias + scale).tolist(),
        }

    @classmethod
    def from_export_cfg(cls, cfg: dict, device="cpu"):
        from gymnasium.spaces import Box

        low = np.asarray(cfg["action_low"], dtype=np.float32)
        high = np.asarray(cfg["action_high"], dtype=np.float32)
        seq_len = cfg.get("seq_len")
        raw_obs_dim = int(cfg.get("raw_obs_dim") or 10)
        hidden_dim = list(cfg["hidden_dim"])
        action_dim = int(cfg["action_dim"])
        state_dim = int(seq_len or 1) * raw_obs_dim
        net = cls(
            state_dim,
            hidden_dim,
            action_dim,
            Box(low=low, high=high, dtype=np.float32),
            log_std_init=float(cfg.get("log_std_init", -3.0)),
            use_sde=bool(cfg.get("use_sde", True)),
            clip_mean=float(cfg.get("clip_mean", 2.0)),
            seq_len=seq_len,
            raw_obs_dim=raw_obs_dim,
            gru_hidden_dim=int(cfg.get("gru_hidden_dim") or 64),
            cond_action_dim=int(cfg.get("cond_action_dim") or 0),
            use_h_norm=bool(cfg.get("use_h_norm", False)),
        )
        return net.to(device)


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, use_orthogonal_init=False,
                 seq_len=None, raw_obs_dim=10, gru_hidden_dim=64, cond_action_dim=0,
                 use_layer_norm=False, use_h_norm=False):
        super(QValueNetContinuous, self).__init__()
        self.cond_action_dim = int(cond_action_dim)
        self.use_layer_norm = bool(use_layer_norm)
        self.encoder = None
        if seq_len is not None:
            self.encoder = GRUObsEncoder(seq_len, raw_obs_dim, gru_hidden_dim,
                                         use_h_norm=use_h_norm)

        layers = []
        input_dim = (self.encoder.joint_dim if self.encoder is not None else state_dim) + action_dim
        input_dim += self.cond_action_dim
        for output_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.fc_latent = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim[-1], 1)
        if use_orthogonal_init:
            orthogonal_init(self.fc_latent)
            orthogonal_init(self.fc_out)

    def forward(self, x, a, cond=None):
        if self.encoder is not None:
            x = self.encoder(x)
        if self.cond_action_dim > 0:
            if cond is None:
                raise ValueError("残差 Q 需要 cond=π_H(s)")
            if not torch.is_tensor(cond):
                cond = torch.as_tensor(cond, dtype=x.dtype, device=x.device)
            else:
                cond = cond.to(device=x.device, dtype=x.dtype)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            x = torch.cat([x, cond], dim=1)
        cat = torch.cat([x, a], dim=1)
        h = self.fc_latent(cat)
        if not self.use_layer_norm:
            h = F.relu(h)
        return self.fc_out(h)

class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_space,
                 actor_lr, critic_lr, alpha_lr, tau, gamma,
                 alpha=1.0,
                 target_network_frequency=1,
                 policy_frequency=1,
                 max_steps=1000000,
                 log_std_init=-3,
                 device=torch.device("cuda"),
                 autotune=True,
                 use_sde=True,
                 use_orthogonal_init=False,
                 clip_mean=2.0,
                 alpha_min: float = None,
                 target_entropy: float = None,
                 seq_len=None,
                 raw_obs_dim=10,
                 gru_hidden_dim=64,
                 use_h_norm=False):
        """
        actor_lr: Actor网络学习率, 值太大，策略会剧烈变化，导致训练震荡甚至发散；如果太小，策略收敛极慢，可能陷入局部最优
        critic_lr: Critic网络学习率, 设置 critic_lr 略大于或等于 actor_lr
        alpha_lr: 温度学习率
        tau: 软更新因子, tau 通常设得非常小（如 0.005 或 0.001）。这意味着目标网络每步只向当前网络"挪动" 0.5% 或 0.1%，变化极其平滑。这能极大防止 Q 值过高估计和训练震荡
        gamma: 折扣因子
        use_sde: 是否使用 gSDE (generalized State-Dependent Exploration)
        log_std_init: log_std 初始值（gSDE 模式下）
        clip_mean: 限制 actor 均值输出范围
        target_entropy: 可覆盖默认的 -action_dim；装配等 7D 任务建议传 -0.5*action_dim
        seq_len: 非 None 时启用 GRU 编码器，观测为 (T, raw_obs_dim)
        use_h_norm: 基座输入用 LayerNorm(h)；仅 BC/GRU 模式有效
        """
        self.use_sde = use_sde
        self.action_space = action_space
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.seq_len = None if seq_len is None else int(seq_len)
        self.raw_obs_dim = int(raw_obs_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.clip_mean = float(clip_mean)
        self.log_std_init = float(log_std_init)
        self.use_orthogonal_init = bool(use_orthogonal_init)
        # 定稿 §1.1：基座输入 [s; h_norm]，h_norm = LayerNorm(h)（带仿射，铁律 2）
        self.use_h_norm = bool(use_h_norm)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr

        net_kw = dict(seq_len=seq_len, raw_obs_dim=raw_obs_dim,
                      gru_hidden_dim=gru_hidden_dim, use_h_norm=self.use_h_norm)
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_space, log_std_init=log_std_init,
                                         use_sde=use_sde, use_orthogonal_init=use_orthogonal_init,
                                         clip_mean=clip_mean, **net_kw).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim, use_orthogonal_init, **net_kw).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim, use_orthogonal_init, **net_kw).to(device)
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim, use_orthogonal_init,
                                                   **net_kw).to(device)
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim, use_orthogonal_init,
                                                   **net_kw).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr, eps=1e-5)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.target_network_frequency = target_network_frequency
        self.policy_frequency = policy_frequency
        self.autotune = autotune
        self.update_iteration = 0
        self.alpha_min = alpha_min  # alpha下限，防止温度坍缩到0导致gSDE探索消失
        self.bc_actor = None  # 冻结 BC，供 ρ_π 上的均值钉住
        self.base_actor = None  # 冻住的 π_H（BeTAIL π_BeT / Johannink π_H）
        self.residual_scale = 1.0  # BeTAIL α：执行 a=clip(π_H + α·π_θ)
        self._replay_action = None  # 写入 buffer_r 的动作（残差模式下为 π_θ，不是执行动作）

        if autotune:
            self.target_entropy = -action_dim if target_entropy is None else float(target_entropy)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, eps=1e-5)
        else:
            self.alpha = alpha

    def reset_noise(self, batch_size=1):
        """SB3：采集段开头 ``reset_noise(n_envs)``；``SAC.train`` 里 ``reset_noise()``。"""
        if self.use_sde:
            self.actor.reset_noise(batch_size)

    def _obs_to_batch(self, state):
        """整理观测为网络输入，返回 (tensor[B,...], return_single)。"""
        states = np.asarray(state, dtype=np.float32)
        if self.seq_len is not None:
            if states.ndim == 2 and states.shape[-1] == self.raw_obs_dim:
                states = states[np.newaxis, ...]
                return_single = True
            elif states.ndim == 1:
                states = states.reshape(1, self.seq_len, self.raw_obs_dim)
                return_single = True
            elif states.ndim in (2, 3):
                return_single = False
            else:
                raise ValueError(f"take_action 不支持 shape={states.shape}")
        elif states.ndim == 1:
            states = states[np.newaxis, :]
            return_single = True
        else:
            return_single = False
        t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return t, return_single

    def _current_frame(self, states):
        """残差 / Q 只用当前帧。历史由冻住的 π_H（GRU-BC）压进 â。"""
        return current_frame(states, self.seq_len, self.raw_obs_dim).to(self.device)

    def _residual_obs(self, states):
        """残差网络无 GRU 时切当前帧；π_H 仍吃整段 stack。"""
        if getattr(self.actor, "encoder", None) is not None:
            return states
        return self._current_frame(states)

    def _base_action(self, states):
        """冻住 π_H 的确定性 â。无残差时返回 None。"""
        if self.base_actor is None:
            return None
        with torch.no_grad():
            return self.base_actor.mean_action(states)

    def _compose_executed(self, base, residual):
        """BeTAIL (1)：a = clip(â + α·ã)，再裁到环境动作盒。"""
        composed = base + float(self.residual_scale) * residual
        low = self.actor.action_bias - self.actor.action_scale
        high = self.actor.action_bias + self.actor.action_scale
        return torch.minimum(torch.maximum(composed, low), high)

    def executed_from_residual(self, states, residual):
        """由回放里的 ã 还原执行动作，供 D(s, â+ã)。"""
        to_numpy = not torch.is_tensor(residual)
        if torch.is_tensor(states):
            states_t = states.to(device=self.device, dtype=torch.float32)
        else:
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if torch.is_tensor(residual):
            residual_t = residual.to(device=self.device, dtype=torch.float32)
        else:
            residual_t = torch.as_tensor(residual, dtype=torch.float32, device=self.device)
        if residual_t.dim() == 1:
            residual_t = residual_t.unsqueeze(0)
        executed = self._compose_executed(self._base_action(states_t), residual_t)
        if to_numpy:
            return executed.detach().cpu().numpy()
        return executed

    def take_action(self, state, deterministic=False, base_only=False):
        """环境执行动作。残差：a=clip(π_H+α·π_θ(s,â))；buffer_r 用 replay_action()。

        ``base_only``：DAWN 数据锚定，只跑冻住的 π_H，回放写入 ã=0。
        """
        states, return_single = self._obs_to_batch(state)
        if base_only:
            if self.base_actor is None:
                raise RuntimeError("base_only 需要先 enable_residual")
            with torch.no_grad():
                executed = self.base_actor.mean_action(states)
            executed_np = np.array(executed.detach().cpu().numpy(), dtype=np.float32, copy=True)
            residual_np = np.zeros_like(executed_np, dtype=np.float32)
            if return_single:
                residual_np = residual_np[0]
                executed_np = executed_np[0]
            self._replay_action = residual_np
            return executed_np
        self.actor.eval()
        with torch.no_grad():
            base = self._base_action(states)
            residual, _ = self.actor(
                self._residual_obs(states), deterministic=deterministic, cond=base)
            executed = residual if base is None else self._compose_executed(base, residual)
        self.actor.train()
        residual_np = np.array(residual.detach().cpu().numpy(), dtype=np.float32, copy=True)
        executed_np = np.array(executed.detach().cpu().numpy(), dtype=np.float32, copy=True)
        if return_single:
            residual_np = residual_np[0]
            executed_np = executed_np[0]
        self._replay_action = residual_np
        return executed_np

    def replay_action(self):
        """上一次 take_action 应写入 SAC 回放的动作（残差或原策略输出）。"""
        if self._replay_action is None:
            raise RuntimeError("replay_action() 需在 take_action() 之后调用")
        return self._replay_action

    def calc_target(self, rewards, next_states, dones):
        """ 计算目标Q值 """
        with torch.no_grad():
            next_base = self._base_action(next_states)
            next_obs = self._residual_obs(next_states)
            next_actions, next_log_prob = self.actor(next_obs, cond=next_base)
            q1_value = self.target_critic_1(next_obs, next_actions, cond=next_base)
            q2_value = self.target_critic_2(next_obs, next_actions, cond=next_base)
            next_q_values = torch.min(q1_value, q2_value) - self.alpha * next_log_prob
        td_target = rewards.flatten() + self.gamma * (1 - dones.flatten()) * next_q_values.view(-1)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict, log_info=True):
        info = {}
        self.update_iteration += 1

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

        base = self._base_action(states)
        next_base = self._base_action(next_states)
        obs = self._residual_obs(states)
        next_obs = self._residual_obs(next_states)

        # 计算 actions_pi 和 log_prob
        actions_pi, log_prob = self.actor(obs, cond=base)
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
            alpha_t = self.alpha if torch.is_tensor(self.alpha) else torch.as_tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )

        # 计算 target Q（使用 pre-step alpha）
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_obs, cond=next_base)
            q1_value = self.target_critic_1(next_obs, next_actions, cond=next_base)
            q2_value = self.target_critic_2(next_obs, next_actions, cond=next_base)
            next_q_values = torch.min(q1_value, q2_value) - alpha_t * next_log_prob
        td_target = rewards.flatten() + self.gamma * (1 - dones.flatten()) * next_q_values.view(-1)

        # 计算 critic loss
        critic_1_values = self.critic_1(obs, actions, cond=base).view(-1)
        critic_2_values = self.critic_2(obs, actions, cond=base).view(-1)
        critic_1_loss = F.mse_loss(critic_1_values, td_target)
        critic_2_loss = F.mse_loss(critic_2_values, td_target)
        critic_loss = 0.5 * (critic_1_loss + critic_2_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 5.0)
        self.critic_optimizer.step()

        if self.update_iteration % self.policy_frequency == 0:
            q1_pi = self.critic_1(obs, actions_pi, cond=base)
            q2_pi = self.critic_2(obs, actions_pi, cond=base)
            sac_actor_loss = (alpha_t * log_prob - torch.min(q1_pi, q2_pi)).mean()
            actor_loss = sac_actor_loss
            bc_loss = None
            bc_onpol_loss = None
            bc_coef = float(transition_dict.get("bc_coef", 0.0) or 0.0)
            bc_onpol_coef = float(transition_dict.get("bc_onpol_coef", 0.0) or 0.0)
            e_s = transition_dict.get("expert_states")
            e_a = transition_dict.get("expert_actions")
            if bc_coef > 0.0 and e_s is not None and e_a is not None:
                # ρ_E：MSE(μ(s_E), a_E)
                bc_loss = F.mse_loss(self.actor.mean_action(_to_dev(e_s)), _to_dev(e_a))
                actor_loss = actor_loss + bc_coef * bc_loss
            if bc_onpol_coef > 0.0 and self.bc_actor is not None:
                # ρ_π（replay 中的策略状态）：MSE(μ_θ(s), μ_BC(s))
                with torch.no_grad():
                    a_bc = self.bc_actor.mean_action(states)
                bc_onpol_loss = F.mse_loss(self.actor.mean_action(states), a_bc)
                actor_loss = actor_loss + bc_onpol_coef * bc_onpol_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

        if self.update_iteration % self.target_network_frequency == 0:
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)

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
            if self.update_iteration % self.policy_frequency == 0:
                info["actor_loss"] = actor_loss.item()
                info["actor_sac_loss"] = sac_actor_loss.item()
                if bc_loss is not None:
                    info["bc_loss"] = bc_loss.item()
                    info["bc_term"] = float(bc_coef) * bc_loss.item()
                if bc_onpol_loss is not None:
                    info["bc_onpol_loss"] = bc_onpol_loss.item()
                    info["bc_onpol_term"] = float(bc_onpol_coef) * bc_onpol_loss.item()
            if self.base_actor is not None:
                res_rms = float(actions.pow(2).mean().sqrt().item())
                info["residual_rms"] = res_rms
                info["residual_exec_rms"] = res_rms * float(self.residual_scale)

        return info
    
    def lr_decay(self, steps):
        # 线性衰减，但最低保持初始值的 10%，避免后期学习率趋近于 0 导致性能持续恶化
        alpha = max(0.3, 1.0 - steps / self.max_steps)
        lr_actor_now = self.actor_lr * alpha
        lr_critic_now = self.critic_lr * alpha

        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_actor_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_critic_now
        if self.autotune:
            lr_alpha_now = self.alpha_lr * alpha
            for p in self.log_alpha_optimizer.param_groups:
                p['lr'] = lr_alpha_now

    def _policy_net(self, cond_action_dim=0, use_gru=True):
        seq_len = self.seq_len if use_gru else None
        state_dim = self.state_dim if use_gru else self.raw_obs_dim
        return PolicyNetContinuous(
            state_dim, self.hidden_dim, self.action_dim, self.action_space,
            log_std_init=self.log_std_init, use_sde=self.use_sde,
            use_orthogonal_init=self.use_orthogonal_init, clip_mean=self.clip_mean,
            seq_len=seq_len, raw_obs_dim=self.raw_obs_dim,
            gru_hidden_dim=self.gru_hidden_dim, cond_action_dim=int(cond_action_dim),
            use_h_norm=self.use_h_norm if use_gru else False,
        ).to(self.device)

    def _q_net(self, cond_action_dim=0, use_layer_norm=False, use_gru=True):
        seq_len = self.seq_len if use_gru else None
        state_dim = self.state_dim if use_gru else self.raw_obs_dim
        return QValueNetContinuous(
            state_dim, self.hidden_dim, self.action_dim, self.use_orthogonal_init,
            seq_len=seq_len, raw_obs_dim=self.raw_obs_dim,
            gru_hidden_dim=self.gru_hidden_dim, cond_action_dim=int(cond_action_dim),
            use_layer_norm=bool(use_layer_norm),
            use_h_norm=self.use_h_norm if use_gru else False,
        ).to(self.device)

    def _frozen_copy_like_actor(self):
        src = self.actor
        dst = PolicyNetContinuous(
            self.state_dim, src.hidden_dim, self.action_dim, self.action_space,
            log_std_init=src.log_std_init, use_sde=src.use_sde,
            use_orthogonal_init=False, clip_mean=src.clip_mean,
            seq_len=self.seq_len, raw_obs_dim=self.raw_obs_dim,
            gru_hidden_dim=self.gru_hidden_dim, cond_action_dim=0,
            use_h_norm=self.use_h_norm,
        ).to(self.device)
        dst.eval()
        for p in dst.parameters():
            p.requires_grad_(False)
        if dst.use_sde:
            dst.exploration_mat = None
            dst.exploration_matrices = None
        return dst

    def freeze_actor_as_bc(self):
        """复制当前 actor 权重并冻结，作为 ρ_π 上的 BC 均值目标。

        不能 deepcopy 整网：gSDE 的 ``exploration_mat`` 来自 ``rsample()``，
        不是 graph leaf，PyTorch 会直接报错。
        """
        dst = self._frozen_copy_like_actor()
        dst.load_state_dict(self.actor.state_dict())
        self.bc_actor = dst

    def _rebuild_residual_trainable(self, scale=1.0):
        """残差 π_θ / Q 吃当前帧 s_t 与 â；均值初值 0。历史只在冻住的 π_H 里。

        Critic 加 LayerNorm（DAWN），actor 不加。
        """
        self.residual_scale = float(scale)
        cond = self.action_dim
        self.actor = self._policy_net(cond_action_dim=cond, use_gru=False)
        nn.init.zeros_(self.actor.fc_mu.weight)
        nn.init.zeros_(self.actor.fc_mu.bias)
        self.critic_1 = self._q_net(cond_action_dim=cond, use_layer_norm=True, use_gru=False)
        self.critic_2 = self._q_net(cond_action_dim=cond, use_layer_norm=True, use_gru=False)
        self.target_critic_1 = self._q_net(cond_action_dim=cond, use_layer_norm=True, use_gru=False)
        self.target_critic_2 = self._q_net(cond_action_dim=cond, use_layer_norm=True, use_gru=False)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.critic_lr, eps=1e-5)
        self._replay_action = None
        self.reset_noise()

    def enable_residual(self, scale=1.0):
        """冻 BC 为 π_H，残差 π_θ(s_t, â)，执行 a=clip(π_H + α·π_θ)。

        与 BeTAIL 相同：历史只在基策略里，残差是吃 (当前观测, â) 的轻量修正。
        须在 load_policy（BC）之后调用。
        """
        if self.base_actor is not None:
            raise RuntimeError("enable_residual 已调用，不能把当前残差再冻成 π_H")
        self.freeze_actor_as_bc()
        self.base_actor = self.bc_actor
        self._rebuild_residual_trainable(scale)

    def load_policy(self, model_dir):
        _model_dir = Path(model_dir)
        cfg = load_policy_cfg(_model_dir)
        if has_weight(_model_dir, "base_policy_net"):
            dst = self._frozen_copy_like_actor()
            dst.load_state_dict(load_state_dict(_model_dir, "base_policy_net", map_location=self.device))
            self.base_actor = dst
            self.bc_actor = dst
            scale = float(self.residual_scale)
            if cfg is not None and cfg.get("residual_scale") is not None:
                scale = float(cfg["residual_scale"])
            self._rebuild_residual_trainable(scale)
            self.actor.load_state_dict(load_state_dict(_model_dir, "policy_net", map_location=self.device))
            return
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
        if self.base_actor is not None:
            save_state_dict(self.base_actor.state_dict(), _model_dir, "base_policy_net")
        save_state_dict(self.critic_1.state_dict(), _model_dir, "qvalue_net1")
        save_state_dict(self.critic_2.state_dict(), _model_dir, "qvalue_net2")
        cfg = self.actor.export_cfg()
        cfg["residual"] = self.base_actor is not None
        if self.base_actor is not None:
            cfg["residual_scale"] = float(self.residual_scale)
            cfg["critic_layer_norm"] = bool(getattr(self.critic_1, "use_layer_norm", False))
        save_policy_cfg(cfg, _model_dir)


class BehaviorClone:
    def __init__(self, state_dim, action_dim, hidden_dim, epochs, lr, weight_decay, action_space, device,
                log_std_init=-3, use_sde=True, use_orthogonal_init=False, clip_mean=2.0,
                seq_len=None, raw_obs_dim=10, gru_hidden_dim=64, use_h_norm=False):
        self.device = device
        self.use_sde = use_sde
        self.max_epochs = epochs

        self.policy = PolicyNetContinuous(state_dim,
                                          hidden_dim,
                                          action_dim,
                                          action_space,
                                          log_std_init,
                                          use_sde,
                                          use_orthogonal_init,
                                          clip_mean,
                                          seq_len=seq_len,
                                          raw_obs_dim=raw_obs_dim,
                                          gru_hidden_dim=gru_hidden_dim,
                                          use_h_norm=use_h_norm).to(device)

        self.lr = lr
        # BC 只拟合均值；不要用 weight_decay 把 log_std 拉走（SAC 还要用）
        policy_params = [p for n, p in self.policy.named_parameters() if n != "log_std"]
        adamw_kw = dict(lr=lr, eps=1e-5, weight_decay=weight_decay)
        if torch.device(device).type == "cuda":
            try:
                self.optimizer = torch.optim.AdamW(policy_params, fused=True, **adamw_kw)
            except (TypeError, RuntimeError):
                self.optimizer = torch.optim.AdamW(policy_params, **adamw_kw)
        else:
            self.optimizer = torch.optim.AdamW(policy_params, **adamw_kw)

    def take_action(self, states):
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            mu = self.policy.mean_action(states)
        self.policy.train()
        return mu.squeeze(0).detach().cpu().numpy()

    def _batch_to_device(self, states, actions):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        return states, actions

    def validation(self, states, actions):
        states, actions = self._batch_to_device(states, actions)
        self.policy.eval()
        with torch.no_grad():
            loss = F.mse_loss(self.policy.mean_action(states), actions)
        return loss.detach()

    def update(self, states, actions):
        states, actions = self._batch_to_device(states, actions)
        self.policy.train()
        loss = F.mse_loss(self.policy.mean_action(states), actions)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def fit_epoch(self, states, actions, batch_size, drop_last=True):
        """一整轮 BC 更新，每 epoch 只同步一次 GPU。"""
        n = states.shape[0]
        bs = int(batch_size)
        perm = torch.randperm(n, device=states.device)
        end = (n // bs) * bs if drop_last else n
        self.policy.train()
        total = torch.zeros((), device=states.device)
        n_batches = 0
        for i in range(0, end, bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(self.policy.mean_action(states[idx]), actions[idx])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total = total + loss.detach()
            n_batches += 1
        return float(total / max(n_batches, 1))

    def eval_epoch(self, states, actions):
        """验证集一次性前向。"""
        self.policy.eval()
        with torch.no_grad():
            loss = F.mse_loss(self.policy.mean_action(states), actions)
        return float(loss)

    def lr_decay(self, epoch):
        # BC 线性衰减，最低保持初始值的 10%
        alpha = max(0.1, 1.0 - epoch / self.max_epochs)
        lr_now = self.lr * alpha
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def load_model(self, model_dir):
        _model_dir = Path(model_dir)
        self.policy.load_state_dict(load_state_dict(_model_dir, "policy_net", map_location=self.device))

    def save_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        save_state_dict(self.policy.state_dict(), _model_dir, "policy_net")
        save_policy_cfg(self.policy.export_cfg(), _model_dir)
        
        
        
        
        
        