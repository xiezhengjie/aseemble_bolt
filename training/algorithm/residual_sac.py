"""定稿版残差 SAC：冻结 GRU+BC 基座，只训残差 actor/critic。

对应《插装任务残差 SAC-GAIL 算法设计定稿》§1/§3/§5/§6：
  - 铁律 1/2：GRU 与 LayerNorm 随基座在 RL 阶段整体冻结，永不再训；
    h 由冻结 GRU 现算，buffer 里存的 h 永不过期。
  - Actor：输入 (s, h_norm)（三模块同一表示），输出 ã = tanh(θ)·ε，有界残差。
  - Critic：Q(s, h_norm, ã)，熵 / log-prob 全算在 ã 上（BeTAIL）。
  - 执行动作 a^e = clip(a_base + ã, −1, 1)；环境执行与 D 奖励评估用 a^e，
    buffer 存 clip 后的 a^e；采样时恢复 ã = a^e − BC(s_t, h_t)。
  - 成功奖励 R_succ 进入 Bellman target；成功终止不 bootstrap（吸收态处理）。
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path

from algorithm.sac import PolicyNetContinuous, current_frame
from utils.checkpoint import (
    has_weight, load_policy_cfg, load_state_dict, save_policy_cfg, save_state_dict,
)

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def residual_target_entropy(action_dim, residual_scale):
    """残差动作 ã=ε·tanh(x) 上的 SAC 目标熵。

    标准启发式对 y∈[-1,1] 取 H*(y)=−d；测度变换 H(ã)≈H(y)+d·log(ε)，故
    H*(ã)= −d + d·log(ε)。ε=0.05、d=6 → ≈−24.0（而非错误的 −6）。
    误用 −d 时目标高于残差盒可达熵上界，α 会单调上升。
    """
    d = float(action_dim)
    eps = float(residual_scale)
    if eps <= 0.0:
        raise ValueError(f"residual_scale 必须 > 0，当前 {residual_scale}")
    return -d + d * float(np.log(eps))


class ResidualActor(nn.Module):
    """残差策略 π_θ(·|s, h_norm)：Gaussian 均值经 tanh 压到 [−ε, ε]。

    与定稿 §3 一致：只用残差动作视图；熵/log-prob 全算在 ã 上。
    均值头零初始化（初始 ã≈0，即"残差为零退化为 BC"）。
    """

    def __init__(self, s_dim, h_dim, action_dim, hidden_dim=(512, 512),
                 residual_scale=0.05, log_std_init=-3.0):
        super().__init__()
        self.s_dim = int(s_dim)
        self.h_dim = int(h_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = list(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.log_std_init = float(log_std_init)

        layers = []
        input_dim = self.s_dim + self.h_dim
        for output_dim in self.hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.trunk = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(input_dim, action_dim)
        self.fc_log_std = nn.Linear(input_dim, action_dim)
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.constant_(self.fc_log_std.weight, 0.0)
        nn.init.constant_(self.fc_log_std.bias, log_std_init)

    def forward(self, s, h_norm):
        x = torch.cat([s, h_norm], dim=1)
        z = self.trunk(x)
        mu = self.fc_mu(z)
        log_std = torch.clamp(self.fc_log_std(z), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # ã = tanh(θ) × ε；log-prob 按标准 SAC tanh 修正，但都在残差尺度上
        action = y_t * self.residual_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(
            self.residual_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def mean_residual(self, s, h_norm):
        x = torch.cat([s, h_norm], dim=1)
        mu = self.fc_mu(self.trunk(x))
        return torch.tanh(mu) * self.residual_scale

    def export_cfg(self):
        return {
            "s_dim": self.s_dim,
            "h_dim": self.h_dim,
            "action_dim": self.action_dim,
            "hidden_dim": list(self.hidden_dim),
            "residual_scale": float(self.residual_scale),
            "log_std_init": float(self.log_std_init),
        }


class ResidualCritic(nn.Module):
    """Q(s, h_norm, ã)：残差动作视图（定稿 §3 表）。"""

    def __init__(self, s_dim, h_dim, action_dim, hidden_dim=(512, 512)):
        super().__init__()
        layers = []
        input_dim = int(s_dim) + int(h_dim) + int(action_dim)
        for output_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.trunk = nn.Sequential(*layers)
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, s, h, a_res):
        x = torch.cat([s, h, a_res], dim=1)
        return self.fc_out(self.trunk(x)).squeeze(-1)


class ResidualSAC:
    """冻结基座 + 残差 SAC 主体（定稿 §1/§3/§6/§7）。

    基座（GRU + LayerNorm + BC 头）来自 ``BehaviorClone`` 训练产物，
    加载后全部冻结（铁律 1/2）；残差模块输入统一用 h_norm（定稿 §5）。
    """

    def __init__(self, base_model_dir, raw_obs_dim=10, action_dim=6,
                 seq_len=8, gru_hidden_dim=64, action_low=None, action_high=None,
                 hidden_dim=(512, 512), residual_scale=0.05,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-4,
                 gamma=0.99, tau=0.005, alpha=1.0, alpha_min=0.12,
                 alpha_max=1.0, target_entropy=None, device=torch.device("cpu"),
                 target_network_frequency=2, policy_frequency=1,
                 actor_ema_tau=0.995):
        self.device = device
        self.raw_obs_dim = int(raw_obs_dim)
        self.action_dim = int(action_dim)
        self.seq_len = int(seq_len)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.hidden_dim = list(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.base_model_dir = Path(base_model_dir)
        # 更新频率（基准 train_sac_gail_ur5e.py：target=2、policy=1）
        self.target_network_frequency = int(target_network_frequency)
        self.policy_frequency = int(policy_frequency)
        self.actor_ema_tau = float(actor_ema_tau)
        self.actor_ema = None

        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.alpha_lr = float(alpha_lr)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.initial_alpha = float(alpha)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.update_iteration = 0

        # ---------- 基座：GRU + LayerNorm + BC 头，加载即冻结 ----------
        # 无 policy_cfg.json 时从 state_dict 反推（含 use_h_norm）
        cfg = load_policy_cfg(self.base_model_dir)
        if not cfg and has_weight(self.base_model_dir, "policy_net"):
            from utils.checkpoint import cfg_from_state_dict
            cfg = cfg_from_state_dict(
                load_state_dict(self.base_model_dir, "policy_net", map_location="cpu"),
                seq_len=self.seq_len)
            print(f"[ResidualSAC] 无 policy_cfg.json，从权重反推 use_h_norm={cfg.get('use_h_norm')}")
        cfg = cfg or {}
        self.base_hidden_dim = list(cfg.get("hidden_dim") or [512, 512])
        low = np.asarray(action_low if action_low is not None
                         else cfg.get("action_low", [-1.0] * action_dim), dtype=np.float32)
        high = np.asarray(action_high if action_high is not None
                          else cfg.get("action_high", [1.0] * action_dim), dtype=np.float32)
        self.base_cfg = {
            "hidden_dim": self.base_hidden_dim,
            "action_dim": self.action_dim,
            "seq_len": int(cfg.get("seq_len") or self.seq_len),
            "raw_obs_dim": int(cfg.get("raw_obs_dim") or self.raw_obs_dim),
            "gru_hidden_dim": int(cfg.get("gru_hidden_dim") or self.gru_hidden_dim),
            "use_h_norm": bool(cfg.get("use_h_norm", False)),
            "action_low": low.tolist(),
            "action_high": high.tolist(),
        }
        self.action_low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.action_high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
        self._base_export_extras = {
            "clip_mean": float(cfg.get("clip_mean", 2.0)),
            "log_std_init": float(cfg.get("log_std_init", -3.67)),
            "use_sde": bool(cfg.get("use_sde", True)),
        }
        self.base_actor = PolicyNetContinuous.from_export_cfg(
            dict(self.base_cfg, **self._base_export_extras), device=device)
        self._load_base_weights()
        self.freeze_base()

        # ---------- 残差模块 ----------
        h_dim = self.base_actor.encoder.norm_dim
        self.actor = ResidualActor(
            self.raw_obs_dim, h_dim, self.action_dim, self.hidden_dim,
            residual_scale=self.residual_scale).to(device)
        # v9：评估/部署用 Polyak-EMA actor。在线 actor 继续探索学习；
        # 确定性评估读 EMA 权重，阻尼 Q 地貌微调引起的策略抖动。
        if self.actor_ema_tau > 0.0:
            self.actor_ema = copy.deepcopy(self.actor)
            for p in self.actor_ema.parameters():
                p.requires_grad_(False)
        self.critic_1 = ResidualCritic(self.raw_obs_dim, h_dim, self.action_dim,
                                       self.hidden_dim).to(device)
        self.critic_2 = ResidualCritic(self.raw_obs_dim, h_dim, self.action_dim,
                                       self.hidden_dim).to(device)
        self.target_critic_1 = ResidualCritic(self.raw_obs_dim, h_dim, self.action_dim,
                                              self.hidden_dim).to(device)
        self.target_critic_2 = ResidualCritic(self.raw_obs_dim, h_dim, self.action_dim,
                                              self.hidden_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.critic_lr, eps=1e-5)

        self.autotune = True
        # 熵在 ã=ε·tanh(x) 上计算；目标必须带 d·log(ε)，不能照搬满量程 −d
        if target_entropy is None:
            self.target_entropy = residual_target_entropy(
                self.action_dim, self.residual_scale)
        else:
            self.target_entropy = float(target_entropy)
        self.log_alpha = torch.tensor(
            np.log(self.initial_alpha), dtype=torch.float32,
            requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, eps=1e-5)
        self.alpha = self.log_alpha.exp().item()
        print(
            f"[ResidualSAC] ε={self.residual_scale:g} target_H={self.target_entropy:.3f} "
            f"(naive -dim={-self.action_dim:g}) α∈[{self.alpha_min},{self.alpha_max}] "
            f"actor_ema={self.actor_ema_tau:g}"
        )

    # ---------- 基座冻结与推理（铁律 1/2） ----------
    def _rebuild_base_actor(self):
        self.base_actor = PolicyNetContinuous.from_export_cfg(
            dict(self.base_cfg, **self._base_export_extras), device=self.device)

    def _load_base_weights(self):
        if not has_weight(self.base_model_dir, "policy_net"):
            raise FileNotFoundError(f"基座 BC 权重不存在: {self.base_model_dir}")
        state = load_state_dict(self.base_model_dir, "policy_net", map_location=self.device)
        state_has_h_norm = any(k.startswith("encoder.h_norm.") for k in state)
        model_has_h_norm = self.base_actor.encoder.h_norm is not None
        # 与权重对齐：有键则开 LayerNorm，无键则关
        if state_has_h_norm != model_has_h_norm:
            self.base_cfg["use_h_norm"] = bool(state_has_h_norm)
            print(f"[ResidualSAC] 按权重对齐 use_h_norm={self.base_cfg['use_h_norm']}")
            self._rebuild_base_actor()
        self.base_actor.load_state_dict(state)

    def freeze_base(self):
        """冻结 GRU + LayerNorm + BC 头（RL 阶段只推理，铁律 1/2）。"""
        self.base_actor.eval()
        for p in self.base_actor.parameters():
            p.requires_grad_(False)

    def base_hidden(self, states, normalized=True):
        """整段叠帧观测 → h_norm（冻结 GRU + 冻结 LayerNorm）。"""
        return self.base_actor.encoder.hidden_only(states, normalized=normalized)

    def base_hidden_batch(self, states):
        """numpy (N, T, D) / (N, T*D) → h_norm numpy (N, H)，供收集器与 buffer。"""
        t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        with torch.no_grad():
            h = self.base_hidden(t, normalized=True)
        return h.cpu().numpy()

    def base_action(self, states, h_norm=None):
        """a_base = BC(s, h_norm)（当前帧 + 头路径，不重算 GRU）。"""
        if h_norm is None:
            h_norm = self.base_hidden(states)
        s = current_frame(states, self.seq_len, self.raw_obs_dim)
        return self.base_actor.mean_action_from_hidden(s, h_norm)

    # ---------- 执行与回放视图（定稿 §3） ----------
    def _clip_executed(self, a_exec):
        return torch.clamp(a_exec, self.action_low_t, self.action_high_t)

    def _as_states_tensor(self, states):
        t = np.asarray(states, dtype=np.float32)
        return torch.as_tensor(t, dtype=torch.float32, device=self.device)

    def take_action(self, states, deterministic=False):
        """环境执行动作 a^e = clip(a_base + ã, −1, 1)。

        兼容 ``rollout_eval``：单条 (T, D) 进、(A,) 出；批量 (B, T, D) 进、(B, A) 出。
        """
        single = np.asarray(states).ndim == 2
        states_t = self._as_states_tensor(np.asarray(states)[None] if single else states)
        with torch.no_grad():
            h_norm = self.base_hidden(states_t)
            a_base = self.base_action(states_t, h_norm)
            s = self._s_frame(states_t)
            if deterministic:
                eval_actor = self.actor_ema if self.actor_ema is not None else self.actor
                a_res = eval_actor.mean_residual(s, h_norm)
            else:
                a_res, _ = self.actor(s, h_norm)
            a_exec = self._clip_executed(a_base + a_res)
        out = a_exec.cpu().numpy()
        return out[0] if single else out

    def act_collect(self, states):
        """采集用：返回 (a^e, ã, h_norm)，三者均 numpy，供 buffer_r 与日志。"""
        states_t = self._as_states_tensor(states)
        with torch.no_grad():
            h_norm = self.base_hidden(states_t)
            a_base = self.base_action(states_t, h_norm)
            s = self._s_frame(states_t)
            a_res, _ = self.actor(s, h_norm)
            a_exec = self._clip_executed(a_base + a_res)
        return (a_exec.cpu().numpy(), a_res.cpu().numpy(), h_norm.cpu().numpy())

    def take_action_base_only(self, states):
        """预填充用：只跑冻结 BC（ã=0），返回 (a^e, h_norm)。"""
        states_t = self._as_states_tensor(states)
        with torch.no_grad():
            h_norm = self.base_hidden(states_t)
            a_base = self.base_action(states_t, h_norm)
            a_exec = self._clip_executed(a_base)
        return a_exec.cpu().numpy(), h_norm.cpu().numpy()

    def executed_from_residual(self, states, h_norm, a_res):
        """从回放的 ã 还原执行动作（供诊断；正常路径直接存 a^e）。"""
        with torch.no_grad():
            a_base = self.base_action(states, h_norm)
            return self._clip_executed(a_base + a_res)

    def recover_residual(self, states, h_norm, a_exec):
        """采样时恢复 ã = a^e − BC(s_t, h_t)（BC 冻结，零成本，定稿 §3）。

        clip 触发时恢复出的是"实际驱动转移的有效残差"，给 critic 拟合正确。
        """
        with torch.no_grad():
            a_base = self.base_action(states, h_norm)
        return a_exec - a_base

    # ---------- 训练（定稿 §6/§7） ----------
    def _s_frame(self, states):
        return current_frame(states, self.seq_len, self.raw_obs_dim)

    def update(self, transition_dict, log_info=True):
        info = {}
        self.update_iteration += 1

        def _to_dev(x):
            t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
            return t

        # (B, T, D) 叠帧观测 → 当前帧 + h_norm
        states = _to_dev(transition_dict["states"])
        next_states = _to_dev(transition_dict["next_states"])
        s = self._s_frame(states)
        s_next = self._s_frame(next_states)
        h = _to_dev(transition_dict["h"])            # buffer 存的 h_t（冻结 GRU 现算）
        h_next = _to_dev(transition_dict["h_next"])  # h_{t+1}
        a_exec = _to_dev(transition_dict["actions"])  # clip 后的 a^e
        dones = _to_dev(transition_dict["dones"]).flatten()
        successes = _to_dev(transition_dict.get("successes", np.zeros_like(
            transition_dict["dones"], dtype=np.float32))).flatten()
        rewards = _to_dev(transition_dict["rewards"]).flatten()  # r̃ 现算 + R_succ

        # 恢复有效残差 ã = a^e − a_base（定稿 §3：SAC 只用残差动作视图）
        a_res = self.recover_residual(states, h, a_exec)

        # α 先取 pre-step 值，供 target Q / actor 用
        if self.autotune:
            alpha_t = self.log_alpha.exp().detach()
        else:
            alpha_t = torch.as_tensor(self.alpha, dtype=torch.float32, device=self.device)

        # ---------- critic：y = r̃ + R_succ·1{success} + γ(1−done)[Q_targ − α log f(ã′)] ----------
        with torch.no_grad():
            a_next, log_prob_next = self.actor(s_next, h_next)
            q1_t = self.target_critic_1(s_next, h_next, a_next)
            q2_t = self.target_critic_2(s_next, h_next, a_next)
            next_q = torch.min(q1_t, q2_t) - alpha_t * log_prob_next.reshape(-1)
            # 成功/失败（terminated）：y = r̃ + R_succ·1{succ}（不 bootstrap）；
            # 超时（truncated）：按无限视野正常 bootstrap（v13 修复：
            # 不再把超时当终点，否则孔区被伪终点零价值淹没，Pardo 2018）。
            td_target = rewards + self.gamma * (1.0 - dones) * next_q

        q1 = self.critic_1(s, h, a_res)
        q2 = self.critic_2(s, h, a_res)
        # Huber(β=10)：R_succ 量级（~10^3）的 bootstrap 尖峰转入线性区，
        # 避免稠密 r̃ 的拟合信号在 MSE+全局 grad-clip 下被成功样本淹没
        critic_loss = F.smooth_l1_loss(q1, td_target, beta=10.0) \
            + F.smooth_l1_loss(q2, td_target, beta=10.0)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 5.0)
        self.critic_optimizer.step()

        # ---------- actor + α（熵/log-prob 全算在 ã 上） ----------
        if self.update_iteration % self.policy_frequency == 0:
            a_pi, log_prob = self.actor(s, h)
            q1_pi = self.critic_1(s, h, a_pi)
            q2_pi = self.critic_2(s, h, a_pi)
            actor_loss = (alpha_t * log_prob.reshape(-1) - torch.min(q1_pi, q2_pi)).mean()

            if self.autotune:
                alpha_loss = -(self.log_alpha *
                               (log_prob.reshape(-1) + self.target_entropy).detach()).mean()
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                with torch.no_grad():
                    lo = (float(np.log(self.alpha_min))
                          if self.alpha_min is not None else None)
                    hi = (float(np.log(self.alpha_max))
                          if self.alpha_max is not None else None)
                    if lo is not None and hi is not None:
                        self.log_alpha.clamp_(min=lo, max=hi)
                    elif lo is not None:
                        self.log_alpha.clamp_(min=lo)
                    elif hi is not None:
                        self.log_alpha.clamp_(max=hi)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            if self.actor_ema is not None:
                with torch.no_grad():
                    for p_ema, p in zip(self.actor_ema.parameters(),
                                        self.actor.parameters()):
                        p_ema.mul_(self.actor_ema_tau).add_(
                            p.detach(), alpha=1.0 - self.actor_ema_tau)

        if self.update_iteration % self.target_network_frequency == 0:
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)

        if log_info:
            self.alpha = self.log_alpha.exp().item() if self.autotune else self.alpha
            info = {
                "critic_loss": float(critic_loss.item()),
                "critic_q1": float(q1.mean().item()),
                "critic_q2": float(q2.mean().item()),
                "alpha": float(self.alpha),
                "residual_rms": float(a_res.pow(2).mean().sqrt().item()),
            }
            if self.update_iteration % self.policy_frequency == 0:
                info["actor_loss"] = float(actor_loss.item())
                if self.autotune:
                    info["alpha_loss"] = float(alpha_loss.item())
        return info

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def lr_decay(self, steps, max_steps=2_000_000):
        alpha = max(0.3, 1.0 - steps / max(1, int(max_steps)))
        for p in self.actor_optimizer.param_groups:
            p["lr"] = self.actor_lr * alpha
        for p in self.critic_optimizer.param_groups:
            p["lr"] = self.critic_lr * alpha
        for p in self.log_alpha_optimizer.param_groups:
            p["lr"] = self.alpha_lr * alpha

    # ---------- 存取 ----------
    def export_cfg(self):
        cfg = {
            "residual": True,
            "residual_scale": float(self.residual_scale),
            "hidden_dim": list(self.hidden_dim),
            "raw_obs_dim": self.raw_obs_dim,
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
            "gru_hidden_dim": self.gru_hidden_dim,
            "base_model_dir": str(self.base_model_dir),
            "use_h_norm": bool(self.base_cfg.get("use_h_norm", False)),
        }
        return cfg

    def save_model(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        policy_state = (self.actor_ema.state_dict()
                        if self.actor_ema is not None else self.actor.state_dict())
        save_state_dict(policy_state, model_dir, "policy_net")
        save_state_dict(self.critic_1.state_dict(), model_dir, "qvalue_net1")
        save_state_dict(self.critic_2.state_dict(), model_dir, "qvalue_net2")
        save_state_dict(self.log_alpha.detach().cpu(), model_dir, "log_alpha")
        # 基座权重一并落盘：checkpoint 自包含，部署侧无需再找 BC 目录
        save_state_dict(self.base_actor.state_dict(), model_dir, "base_policy_net")
        save_policy_cfg(self.export_cfg(), model_dir)

    def load_model(self, model_dir):
        model_dir = Path(model_dir)
        policy_state = load_state_dict(model_dir, "policy_net",
                                       map_location=self.device)
        self.actor.load_state_dict(policy_state)
        if self.actor_ema is not None:
            self.actor_ema.load_state_dict(policy_state)
        self.critic_1.load_state_dict(load_state_dict(model_dir, "qvalue_net1",
                                                      map_location=self.device))
        self.critic_2.load_state_dict(load_state_dict(model_dir, "qvalue_net2",
                                                      map_location=self.device))
        if has_weight(model_dir, "log_alpha"):
            self.log_alpha.data.copy_(load_state_dict(model_dir, "log_alpha").to(
                self.device))
        # 自包含 checkpoint：基座权重随存随取；缺省回退构造时的 BC 目录
        if has_weight(model_dir, "base_policy_net"):
            self.base_actor.load_state_dict(load_state_dict(
                model_dir, "base_policy_net", map_location=self.device))
            self.freeze_base()
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
