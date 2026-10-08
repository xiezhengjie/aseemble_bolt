import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from training.common.rl_utils import orthogonal_init, layer_init

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


class PolicyNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, action_space,
                 log_std_init=-3.0, use_sde=True, clip_mean=2.0):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = list(hidden_dim)
        self.use_sde = use_sde
        self.log_std_init = log_std_init
        self.clip_mean = clip_mean

        layers=[]
        input_dim = int(state_dim)
        for output_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.SiLU())
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
        exploration_matrices = self.exploration_matrices.to(device=latent_sde.device, dtype=latent_sde.dtype)
        n_batch = latent_sde.shape[0]
        if n_batch == 1 or n_batch != exploration_matrices.shape[0]:
            return torch.mm(latent_sde, exploration_mat)
        return torch.bmm(latent_sde.unsqueeze(1), exploration_matrices).squeeze(1)


    def _latent(self, x):
        return self.fc_latent(x)


    def forward(self, x, deterministic=False):
        latent = self._latent(x)
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

    def mean_action(self, x):
        """确定性均值动作（BC / eval），不走 gSDE 与 log_prob。"""
        latent = self._latent(x)
        mu = self.mu_clamp(self.fc_mu(latent))
        return torch.tanh(mu) * self.action_scale + self.action_bias

    def export_cfg(self) -> dict:
        return {
            "hidden_dim": list(self.hidden_dim),
            "action_dim": int(self.action_dim),
            "raw_obs_dim": int(self.fc_latent[0].in_features ),
            "clip_mean": float(self.clip_mean),
            "log_std_init": float(self.log_std_init),
            "use_sde": bool(self.use_sde),
        }

    @classmethod
    def from_export_cfg(cls, cfg: dict, device="cpu"):
        from gymnasium.spaces import Box

        low = np.asarray(cfg["action_low"], dtype=np.float32)
        high = np.asarray(cfg["action_high"], dtype=np.float32)
        hidden_dim = list(cfg["hidden_dim"])
        action_dim = int(cfg["action_dim"])
        raw_obs_dim = int(cfg.get("raw_obs_dim") or 10)
        state_dim = raw_obs_dim
        net = cls(
            state_dim,
            hidden_dim,
            action_dim,
            Box(low=low, high=high, dtype=np.float32),
            log_std_init=float(cfg.get("log_std_init", -3.0)),
            use_sde=bool(cfg.get("use_sde", True)),
            clip_mean=float(cfg.get("clip_mean", 2.0)),
        )
        return net.to(device)



class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        layers = []
        input_dim = int(state_dim) + action_dim
        for output_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.SiLU())
            input_dim = output_dim
        self.fc_latent = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim[-1], 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        h = self.fc_latent(cat)
        return self.fc_out(h)

