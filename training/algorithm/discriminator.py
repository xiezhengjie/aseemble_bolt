import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from utils.rl_utils import orthogonal_init
from algorithm.sac import GRUObsEncoder, current_frame

DISC_LOGIT_CLAMP = 20.0  # z clamp ±20 → softplus(−z)∈[≈0,20]；对外 r̃/20 ∈[0,1]

def _r1_gradient_penalty(disc_out, inputs, coef=10.0):
    """BeTAIL 原配 R1 梯度惩罚：10.0 × ‖∇_x D‖²（对 sigmoid 输出的 logit 加权）。

    disc_out: D 网对 (s,a) 的 logit z；inputs: 计算 z 时实际消费的输入张量元组
    （必须都在 disc_out 的计算图里，不能传入事后拼接的副本）。
    """
    grads = torch.autograd.grad(
        outputs=torch.sigmoid(disc_out).sum(), inputs=inputs,
        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True,
    )
    gp = torch.zeros((), device=disc_out.device)
    for g in grads:
        if g is not None:
            gp = gp + (g.pow(2).reshape(g.shape[0], -1).sum(dim=1)).mean()
    return float(coef) * 0.5 * gp


def _disc_entropy(logit):
    """判别器输出分布的（二值)熵 H(D_φ)，用于熵正则防过自信。"""
    p = torch.sigmoid(logit)
    return -(p * torch.log(p + 1e-8) + (1.0 - p) * torch.log(1.0 - p + 1e-8)).mean()


def _linear(in_dim, out_dim, use_orthogonal_init=False, gain=None, use_spectral_norm=False):
    """先正交初始化，再包谱归一化（SN 作用在 weight_orig 上）。"""
    lin = nn.Linear(in_dim, out_dim)
    if use_orthogonal_init:
        orthogonal_init(lin, np.sqrt(2) if gain is None else gain)
    if use_spectral_norm:
        lin = spectral_norm(lin)
    return lin


class DiscriminatorNN(nn.Module):
    """与 QValueNetContinuous 相同：可选 GRUObsEncoder，再 cat(a) 进 MLP。"""

    def __init__(self, state_dim, action_dim, hidden_dim, use_orthogonal_init=True,
                 use_spectral_norm=False, seq_len=None, raw_obs_dim=10, gru_hidden_dim=64):
        super(DiscriminatorNN, self).__init__()
        self.encoder = None
        if seq_len is not None:
            self.encoder = GRUObsEncoder(seq_len, raw_obs_dim, gru_hidden_dim)

        layers = []
        input_dim = (self.encoder.joint_dim if self.encoder is not None else state_dim) + action_dim
        for output_dim in hidden_dim:
            layers.append(_linear(
                input_dim, output_dim,
                use_orthogonal_init=use_orthogonal_init,
                use_spectral_norm=use_spectral_norm,
            ))
            layers.append(nn.ReLU())
            input_dim = output_dim

        self.fc_latent = nn.Sequential(*layers)
        self.fc_out = _linear(
            hidden_dim[-1], 1,
            use_orthogonal_init=use_orthogonal_init,
            gain=0.01,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x, a):
        if self.encoder is not None:
            x = self.encoder(x)
        x = torch.cat([x, a], dim=1)
        x = self.fc_latent(x)
        return self.fc_out(x)


class Discriminator():
    """GAIL 判别器。D = P(policy|s,a)：expert 标签 smoothing，policy 1-smoothing。

    奖励（``reward_form``）：
      - ``logit``：r = clip(-ℓ, -c, c)，即 clip 后的 log((1-D)/D)；D=0.5 时为 0。
      - ``neglogd``：r = -log(D)，再只裁上侧（旧 GAIL，恒正）。
    更新：从 buffer_e / buffer_g 各抽完整轨迹，直到 (s,a) 数 ≥ batch_size。
    ``seq_len`` 非空时观测走与 SAC 相同的 GRUObsEncoder（独立权重），MLP 头不变。
    """
    def __init__(self, state_dim, action_dim, hidden_dim, lr, smoothing, weight_decay,
                 epoch, batch_size, max_steps, device, use_orthogonal_init=True,
                 grad_clip_norm=None,
                 reward_clip: float = None,
                 reward_scale: float = 1.0,
                 use_spectral_norm: bool = False,
                 reward_form: str = "neglogd",
                 seq_len=None, raw_obs_dim=10, gru_hidden_dim=64):
        self.gail_epoch = epoch
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.smoothing = smoothing
        self.lr = lr
        self.max_steps = max_steps
        self.grad_clip_norm = grad_clip_norm
        self.reward_clip = reward_clip
        self.reward_scale = reward_scale
        form = str(reward_form).lower()
        if form not in ("logit", "neglogd"):
            raise ValueError(f"reward_form 须为 logit 或 neglogd，收到 {reward_form!r}")
        self.reward_form = form
        self.disc = DiscriminatorNN(
            state_dim, action_dim, hidden_dim,
            use_orthogonal_init=use_orthogonal_init,
            use_spectral_norm=use_spectral_norm,
            seq_len=seq_len, raw_obs_dim=raw_obs_dim,
            gru_hidden_dim=gru_hidden_dim,
        ).to(device)
        self.disc_optim = torch.optim.AdamW(self.disc.parameters(), lr=lr, eps=1e-5, weight_decay=weight_decay)

    def _as_2d(self, x):
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif self.disc.encoder is None and t.dim() > 2:
            t = t.reshape(t.shape[0], -1)
        return t.to(self.device)

    def update(self, buffer_e, buffer_g, running_ms=None, log_info=True,
               expert_suffix_frac=0.0, expert_suffix_ratio=0.0):
        if buffer_e.n_trajectories() < 1 or buffer_g.n_trajectories() < 1:
            return {}
        info = {
            'loss': [],
            'expert_value': [],
            'policy_value': [],
            'expert_raw': [],
            'policy_raw': [],
            'raw_gap': [],
            'grad_norm': [],
        }
        for _ in range(self.gail_epoch):
            expert_states, expert_actions = buffer_e.sample_trajectories(
                min_steps=self.batch_size,
                suffix_frac=expert_suffix_frac,
                suffix_ratio=expert_suffix_ratio,
            )
            policy_states, policy_actions = buffer_g.sample_trajectories(min_steps=self.batch_size)

            if running_ms is not None:
                expert_states = running_ms.normalize(expert_states)
                policy_states = running_ms.normalize(policy_states)

            expert_d = self.disc(self._as_2d(expert_states), self._as_2d(expert_actions))
            policy_d = self.disc(self._as_2d(policy_states), self._as_2d(policy_actions))

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.full_like(expert_d, self.smoothing))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.full_like(policy_d, 1-self.smoothing))
            gail_loss = expert_loss + policy_loss

            self.disc_optim.zero_grad()
            gail_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                self.disc.parameters(),
                self.grad_clip_norm if self.grad_clip_norm is not None else float('inf')
            )
            self.disc_optim.step()

            e_raw = expert_d.mean().item()
            p_raw = policy_d.mean().item()
            info['loss'].append(gail_loss.item())
            info['expert_value'].append(torch.sigmoid(expert_d).mean().item())
            info['policy_value'].append(torch.sigmoid(policy_d).mean().item())
            info['expert_raw'].append(e_raw)
            info['policy_raw'].append(p_raw)
            info['raw_gap'].append(e_raw - p_raw)
            info['grad_norm'].append(gn.item() if torch.is_tensor(gn) else float(gn))

        for k in ('loss', 'expert_value', 'policy_value', 'expert_raw', 'policy_raw', 'raw_gap', 'grad_norm'):
            info[k] = np.mean(info[k]) if info[k] else 0.0
        return info

    def predict_policy_prob(self, states, actions, to_numpy=True):
        """D = P(policy|s,a) = sigmoid(ℓ)，与 expert_value / policy_value 同一尺度。"""
        with torch.no_grad():
            logit = self.disc(self._as_2d(states), self._as_2d(actions)).squeeze(-1)
            d = torch.sigmoid(logit)
            if to_numpy:
                return d.cpu().numpy()
            return d

    def predict_rewards(self, states, actions, to_numpy=True):
        with torch.no_grad():
            logit = self.disc(self._as_2d(states), self._as_2d(actions)).squeeze(-1)
            if self.reward_form == "logit":
                gail_rewards = -logit
                if self.reward_clip is not None:
                    c = float(self.reward_clip)
                    gail_rewards = torch.clamp(gail_rewards, -c, c)
            else:
                score = torch.clamp(torch.sigmoid(logit), 1e-3, 1.0 - 1e-3)
                gail_rewards = -score.log()
                if self.reward_clip is not None:
                    gail_rewards = torch.clamp(gail_rewards, max=self.reward_clip)
            if self.reward_scale != 1.0:
                gail_rewards = gail_rewards * self.reward_scale
            if to_numpy:
                return gail_rewards.cpu().numpy()
            return gail_rewards

    def lr_decay(self, steps):
        alpha = max(0.1, 1.0 - steps / self.max_steps)
        lr_now = self.lr * alpha
        for p in self.disc_optim.param_groups:
            p['lr'] = lr_now

    def load_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        self.disc.load_state_dict(torch.load(_model_dir/"discriminator_net.pth", weights_only=True))

    def save_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.disc.state_dict(), model_dir/"discriminator_net.pth")


class MarkovDiscriminator:
    """定稿版 GAIL 判别器：反标签（Shen 同款）、Markovian、(s, a^e) 输入。

    约定（定稿 §0/§2）：
      - z = D_φ(s,a^e) 的 logit；D_φ = σ(z) 表示"来自当前策略（生成样本）"。
      - 训练标签：生成样本（buffer_g）→ 1；专家样本（buffer_e）→ 0。
      - 损失 L_D = −E_g[log D_φ] − E_e[log(1 − D_φ)]；
        正则 = 梯度惩罚（默认 ~1.0）+ 可选熵正则（可关）。
      - 奖励 r̃ = softplus(−z) / 20 ∈ [0,1]（永远从 logit 直算，
        不要先 sigmoid 再取 log）；z clamp ±20。
      - 输入只用当前帧 (s_t, a^e_t)，不要 h（避免捷径学习，定稿 §1.1）。

    更新按 transition 均匀采样（定稿 §9）：buffer_e 固定摊平、buffer_g 为
    FIFO 窗口视图，两池 1:1 组批。
    """

    def __init__(self, raw_obs_dim, action_dim, hidden_dim=(64, 64), lr=1e-3,
                 gp_coef=10.0, entropy_coef=0.001, device=torch.device("cpu"),
                 grad_clip_norm=None, logit_clamp=DISC_LOGIT_CLAMP):
        self.raw_obs_dim = int(raw_obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = list(hidden_dim)
        self.lr = float(lr)
        self.gp_coef = float(gp_coef)
        self.entropy_coef = float(entropy_coef)
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.logit_clamp = float(logit_clamp)

        layers = []
        input_dim = self.raw_obs_dim + self.action_dim
        for output_dim in self.hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.disc = nn.Sequential(
            *layers, nn.Linear(input_dim, 1)).to(device)
        self.disc_optim = torch.optim.Adam(self.disc.parameters(), lr=lr, eps=1e-5)

    def _to_tensor(self, x):
        t = np.asarray(x, dtype=np.float32)
        t = t.reshape(t.shape[0], -1) if t.ndim > 2 else (
            t.reshape(1, -1) if t.ndim == 1 else t)
        return torch.as_tensor(t, dtype=torch.float32, device=self.device)

    def _logit(self, states, actions, requires_grad=False):
        s = self._to_tensor(states)
        a = self._to_tensor(actions)
        if requires_grad:
            s.requires_grad_(True)
            a.requires_grad_(True)
        z = self.disc(torch.cat([s, a], dim=1)).squeeze(-1)
        z = torch.clamp(z, -self.logit_clamp, self.logit_clamp)
        return (z, (s, a)) if requires_grad else z

    def update(self, expert_states, expert_actions, gen_states, gen_actions,
               log_info=True):
        """反标签 BCE + GP + 熵正则。四个输入均为当前帧 (s, a^e) 批。"""
        if len(expert_states) == 0 or len(gen_states) == 0:
            return {}
        z_e, (e_s, e_a) = self._logit(expert_states, expert_actions, requires_grad=True)
        z_g, (g_s, g_a) = self._logit(gen_states, gen_actions, requires_grad=True)

        # 专家样本标签 0，生成样本标签 1（反标签约定，定稿 §0）
        loss_e = F.binary_cross_entropy_with_logits(z_e, torch.zeros_like(z_e))
        loss_g = F.binary_cross_entropy_with_logits(z_g, torch.ones_like(z_g))
        # GP：对真实消费过的输入张量求梯度（不能用事后 cat 的副本）
        gp = _r1_gradient_penalty(z_e, (e_s, e_a), coef=self.gp_coef) \
            + _r1_gradient_penalty(z_g, (g_s, g_a), coef=self.gp_coef)
        ent = _disc_entropy(z_e) + _disc_entropy(z_g)
        loss = loss_e + loss_g + gp - self.entropy_coef * ent

        self.disc_optim.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.grad_clip_norm)
        self.disc_optim.step()

        if not log_info:
            return {}
        with torch.no_grad():
            d_e = torch.sigmoid(z_e.detach()).mean().item()
            d_g = torch.sigmoid(z_g.detach()).mean().item()
        # D 准确率：D>0.5 判为生成样本，与标签比对
        acc = 0.5 * (
            (torch.sigmoid(z_g.detach()) > 0.5).float().mean().item()
            + (torch.sigmoid(z_e.detach()) <= 0.5).float().mean().item())
        return {
            "disc_loss": float(loss.item()),
            "disc_loss_bce": float((loss_e + loss_g).item()),
            "disc_gp": float(gp.item()),
            "disc_entropy": float(ent.item()),
            "disc_expert_value": float(1.0 - d_e),   # 展示为 D_GAIL = 1 − D_φ（P(专家)）
            "disc_gen_value": float(d_g),             # D_φ（P(生成)）
            "disc_acc": float(acc),
        }

    def predict_rewards(self, states, actions, to_numpy=True):
        """r̃ = softplus(−z) / logit_clamp ∈ [0,1]（clamp=20）。

        专家样像 → z→−∞ → r̃→1；像当前策略 → r̃→0。
        """
        with torch.no_grad():
            s = self._to_tensor(states)
            a = self._to_tensor(actions)
            z = self.disc(torch.cat([s, a], dim=1)).squeeze(-1)
            z = torch.clamp(z, -self.logit_clamp, self.logit_clamp)
        r = F.softplus(-z) / float(self.logit_clamp)
        if to_numpy:
            return r.cpu().numpy()
        return r

    def lr_decay(self, steps, max_steps=2_000_000):
        alpha = max(0.1, 1.0 - steps / max_steps)
        lr_now = self.lr * alpha
        for p in self.disc_optim.param_groups:
            p['lr'] = lr_now

    def load_model(self, model_dir):
        from utils.checkpoint import load_state_dict
        self.disc.load_state_dict(load_state_dict(Path(model_dir), "disc_net"))

    def save_model(self, model_dir):
        from utils.checkpoint import save_state_dict
        save_state_dict(self.disc.state_dict(), Path(model_dir), "disc_net")


class MarkovDiscriminatorSS:
    """(s, s′) 型 Markov 判别器：判别"状态演化方式"而非"状态-动作对"。

    动机（实测）：(s,a) 型 D 在"强 BC 初始化 + 局部失败"任务上，会把
    策略与专家最好区分的失败区（孔口）打成最低分，形成奖励谷地，
    稠密奖励与任务进展反相关（spearman(r̃, xy误差)≈+0.7）。
    (s,s′) 型 D 判别的是"这一步状态有没有像专家那样演化"（xy 误差缩小、
    深度增加 vs 停滞/抖动），奖励天然与进展同向，谷地消失。

    约定与 MarkovDiscriminator 相同：
      - z = D_φ(s,s′) 的 logit；D_φ = σ(z) 表示"来自当前策略（生成样本）"。
      - 训练标签：生成样本 → 1；专家样本 → 0（反标签）。
      - 奖励 r̃ = softplus(−z)/clamp ∈ [0,1]；z clamp ±20。
      - 正则：R1 梯度惩罚（默认 1.0）+ 可选熵正则（默认关）。
    输入只用当前帧 s_t 与 s_{t+1}（各 raw_obs_dim 维），不用 h。
    """

    def __init__(self, raw_obs_dim, hidden_dim=(64, 64), lr=1e-3,
                 gp_coef=1.0, entropy_coef=0.0, device=torch.device("cpu"),
                 grad_clip_norm=None, logit_clamp=DISC_LOGIT_CLAMP,
                 reward_logit_clamp=4.0, ema_tau=0.99, reward_binary=True):
        self.raw_obs_dim = int(raw_obs_dim)
        self.hidden_dim = list(hidden_dim)
        self.lr = float(lr)
        self.gp_coef = float(gp_coef)
        self.entropy_coef = float(entropy_coef)
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.logit_clamp = float(logit_clamp)
        # v4：奖励用独立 clamp（±4），不 ÷20 —— ÷20 会把近专家数据的
        # 有效对比度压到 ~0.01/步，被熵项 α·logπ≈1.2/步 淹没（v3 实测封顶）
        self.reward_logit_clamp = float(reward_logit_clamp)
        # v7（SQIL 化）：奖励二值化 r̃ = C·1{z<0}，C=softplus(clamp)。
        # v4-v6 实测：幅度型奖励的幅度携带与任务无关的"区域可分性"信息，
        # 可被务农（v4 离孔务农）且随 D 摆动（v5/v6 极限环）；常数奖励
        # 只保留"支撑集内/外"的边界信息（SQIL, Reddy et al. 2019），
        # 无可务农的幅度差，奖励漂移被限幅到 ±C。
        self.reward_binary = bool(reward_binary)
        # v6：奖励用 EMA 副本产出 —— v5 实测 D 与策略互相追逐形成极限环
        # （D margin 0.52↔0.73 摆动，成功率 50%↔34% 跟随振荡）
        self.ema_tau = float(ema_tau)

        layers = []
        input_dim = 2 * self.raw_obs_dim
        for output_dim in self.hidden_dim:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.disc = nn.Sequential(
            *layers, nn.Linear(input_dim, 1)).to(device)
        self.disc_optim = torch.optim.Adam(self.disc.parameters(), lr=lr, eps=1e-5)
        import copy as _copy
        self.disc_ema = _copy.deepcopy(self.disc)
        for p in self.disc_ema.parameters():
            p.requires_grad_(False)

    def _to_tensor(self, x):
        t = np.asarray(x, dtype=np.float32)
        t = t.reshape(t.shape[0], -1) if t.ndim > 2 else (
            t.reshape(1, -1) if t.ndim == 1 else t)
        return torch.as_tensor(t, dtype=torch.float32, device=self.device)

    def _logit(self, states, next_states, requires_grad=False):
        s = self._to_tensor(states)
        s_n = self._to_tensor(next_states)
        if requires_grad:
            s.requires_grad_(True)
            s_n.requires_grad_(True)
        z = self.disc(torch.cat([s, s_n], dim=1)).squeeze(-1)
        z = torch.clamp(z, -self.logit_clamp, self.logit_clamp)
        return (z, (s, s_n)) if requires_grad else z

    def update(self, expert_states, expert_next, gen_states, gen_next,
               log_info=True):
        """反标签 BCE + GP + 熵正则。输入均为 (s_t, s_{t+1}) 当前帧批。"""
        if len(expert_states) == 0 or len(gen_states) == 0:
            return {}
        z_e, (e_s, e_sn) = self._logit(expert_states, expert_next, requires_grad=True)
        z_g, (g_s, g_sn) = self._logit(gen_states, gen_next, requires_grad=True)

        loss_e = F.binary_cross_entropy_with_logits(z_e, torch.zeros_like(z_e))
        loss_g = F.binary_cross_entropy_with_logits(z_g, torch.ones_like(z_g))
        gp = _r1_gradient_penalty(z_e, (e_s, e_sn), coef=self.gp_coef) \
            + _r1_gradient_penalty(z_g, (g_s, g_sn), coef=self.gp_coef)
        ent = _disc_entropy(z_e) + _disc_entropy(z_g)
        loss = loss_e + loss_g + gp - self.entropy_coef * ent

        self.disc_optim.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.grad_clip_norm)
        self.disc_optim.step()
        with torch.no_grad():
            for p_e, p in zip(self.disc_ema.parameters(), self.disc.parameters()):
                p_e.mul_(self.ema_tau).add_(p.detach(), alpha=1.0 - self.ema_tau)

        if not log_info:
            return {}
        with torch.no_grad():
            d_e = torch.sigmoid(z_e.detach()).mean().item()
            d_g = torch.sigmoid(z_g.detach()).mean().item()
        acc = 0.5 * (
            (torch.sigmoid(z_g.detach()) > 0.5).float().mean().item()
            + (torch.sigmoid(z_e.detach()) <= 0.5).float().mean().item())
        return {
            "disc_loss": float(loss.item()),
            "disc_loss_bce": float((loss_e + loss_g).item()),
            "disc_gp": float(gp.item()),
            "disc_entropy": float(ent.item()),
            "disc_expert_value": float(1.0 - d_e),
            "disc_gen_value": float(d_g),
            "disc_acc": float(acc),
        }

    def predict_rewards(self, states, next_states, to_numpy=True):
        """SQIL 化二值奖励：r̃ = C·1{z_ema<0}，C=softplus(reward_logit_clamp)。

        z<0 ⇔ 被判为"成功向"（正类：专家 ∪ 在线成功回合）。常数幅度
        消除 v4 的务农梯度与 v5/v6 的幅度振荡（SQIL, Reddy et al. 2019）。
        reward_binary=False 时退回 softplus(−z)（v4-v6 幅度型）。
        """
        with torch.no_grad():
            s = self._to_tensor(states)
            s_n = self._to_tensor(next_states)
            z = self.disc_ema(torch.cat([s, s_n], dim=1)).squeeze(-1)
            z = torch.clamp(z, -self.reward_logit_clamp, self.reward_logit_clamp)
        if self.reward_binary:
            c = float(F.softplus(torch.tensor(self.reward_logit_clamp)))
            r = (z < 0).to(torch.float32) * c
        else:
            r = F.softplus(-z)
        if to_numpy:
            return r.cpu().numpy()
        return r

    def lr_decay(self, steps, max_steps=2_000_000):
        alpha = max(0.1, 1.0 - steps / max_steps)
        lr_now = self.lr * alpha
        for p in self.disc_optim.param_groups:
            p['lr'] = lr_now

    def load_model(self, model_dir):
        from utils.checkpoint import load_state_dict
        sd = load_state_dict(Path(model_dir), "disc_net")
        self.disc.load_state_dict(sd)
        self.disc_ema.load_state_dict(sd)

    def save_model(self, model_dir):
        from utils.checkpoint import save_state_dict
        save_state_dict(self.disc_ema.state_dict(), Path(model_dir), "disc_net")


def build_expert_ss_pairs(states, dones, k=1):
    """从专家 npz 构造 (s_t, s_{t+k}) 对：区间 [t, t+k) 内无 done（不跨轨迹）。

    states: (N, raw_obs_dim) 已归一化单帧；dones: (N,)。
    k>1 用 cumsum 技巧保证窗口内无回合终止。
    返回 (s_pairs, s_next_pairs)，均为 (M, raw_obs_dim)。
    """
    states = np.asarray(states, dtype=np.float32)
    dones = np.asarray(dones).reshape(-1)
    n = len(states)
    k = int(k)
    if k <= 1:
        valid = np.where(dones[:-1] == 0)[0]
        return states[valid], states[valid + 1]
    cs = np.concatenate([[0], np.cumsum(dones)])
    t = np.arange(0, n - k)
    valid = t[(cs[t + k] - cs[t]) == 0]
    return states[valid], states[valid + k]
