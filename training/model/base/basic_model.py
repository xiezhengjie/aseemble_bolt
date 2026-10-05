"""普通窗口式行为克隆，使用 MSE 回归示范动作。"""
"""普通窗口式 1D CNN + 空间注意力 + GRU + 时间注意力 + MLP 策略。"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box


def _xavier_init(module):
    """对策略网络的可训练层执行 Xavier 初始化。"""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        if getattr(module, "preserve_identity", False):
            return
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, parameter in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class AttentionPooling(nn.Module):
    """时间注意力池化：对 ``[B, T, D]`` 加权汇总为 ``[B, D]``。"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim or max(1, self.dim // 2))
        self.score = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        # 输入形状为 [B, T, D]
        weights = torch.softmax(self.score(x), dim=1)
        context = (x * weights).sum(dim=1)
        return context, weights


class SpatialAttention1D(nn.Module):
    """对 CNN 特征通道执行初始为恒等映射的残差注意力。"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError("spatial attention 的 channels 必须为正数")
        reduction = int(reduction)
        if reduction < 1:
            raise ValueError("spatial attention reduction 必须为正数")
        reduced = max(1, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(reduced, channels, kernel_size=1),
        )
        # 让注意力在训练开始时严格等于恒等映射，避免连续 sigmoid 门控缩小特征。
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.channel_gate[-1].preserve_identity = True

    def forward(self, features):
        channel_delta = torch.tanh(self.channel_gate(features))
        return features * (1.0 + 0.5 * channel_delta)


class TemporalConvGRUEncoder(nn.Module):
    """沿时间轴的 1D CNN + GRU + 时间注意力编码器。"""

    def __init__(self, seq_len, raw_obs_dim, gru_hidden_dim, dropout=0.2,
                 cnn_channels=32, cnn_kernel_size=3, use_time_attn=False,
                 use_spatial_attn=False, spatial_attn_reduction=4):
        super().__init__()
        self.seq_len = int(seq_len)
        self.raw_obs_dim = int(raw_obs_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.cnn_channels = int(cnn_channels)
        self.cnn_kernel_size = int(cnn_kernel_size)
        self.use_time_attn = bool(use_time_attn)
        self.use_spatial_attn = bool(use_spatial_attn)
        self.spatial_attn_reduction = int(spatial_attn_reduction)
        if self.cnn_channels > 0:
            if self.cnn_kernel_size % 2 != 1:
                raise ValueError("cnn_kernel_size 必须为奇数，以保持序列长度")
            self.cnn = nn.Sequential(
                nn.Conv1d(self.raw_obs_dim, self.cnn_channels,
                          self.cnn_kernel_size, padding=self.cnn_kernel_size // 2),
                nn.BatchNorm1d(self.cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Conv1d(self.cnn_channels, self.cnn_channels,
                          self.cnn_kernel_size, padding=self.cnn_kernel_size // 2),
                nn.BatchNorm1d(self.cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.spatial_attn = (
                SpatialAttention1D(
                    self.cnn_channels,
                    reduction=self.spatial_attn_reduction,
                )
                if self.use_spatial_attn else None
            )
            gru_input_dim = self.cnn_channels
        else:
            self.cnn = None
            self.spatial_attn = None
            gru_input_dim = self.raw_obs_dim
        self.gru = nn.GRU(gru_input_dim, self.gru_hidden_dim, num_layers=2, batch_first=True)
        self.time_attn = (
            AttentionPooling(self.gru_hidden_dim)
            if self.use_time_attn else None
        )
        self.h_norm = nn.LayerNorm(self.gru_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def norm_dim(self):
        return self.gru_hidden_dim

    @property
    def joint_dim(self):
        return self.norm_dim + self.raw_obs_dim

    def hidden_only(self, x, normalized=True):
        """从当前完整窗口提取特征，不保存跨窗口的 GRU 状态。"""
        sequence = as_sequence(x, self.seq_len, self.raw_obs_dim)
        if self.cnn is not None:
            sequence = self.cnn(sequence.transpose(1, 2))
            if self.spatial_attn is not None:
                sequence = self.spatial_attn(sequence)
            sequence = sequence.transpose(1, 2)
        gru_out, hidden = self.gru(sequence)
        pooled = self.time_attn(gru_out)[0] if self.time_attn is not None else hidden[-1]
        if normalized:
            pooled = self.dropout(self.h_norm(pooled))
        return pooled

    def forward(self, x):
        raw_sequence = as_sequence(x, self.seq_len, self.raw_obs_dim)
        hidden = self.hidden_only(raw_sequence)
        return torch.cat([hidden, raw_sequence[:, -1, :]], dim=1)


class BasePolicy(nn.Module):
    """使用确定性动作回归的 BC 基座。"""

    policy_class = "BasePolicy"

    def __init__(self, state_dim, hidden_dim, action_dim, action_space,
                 clip_mean=2.0, seq_len=None, raw_obs_dim=10, gru_hidden_dim=64,
                 cnn_channels=64, cnn_kernel_size=3, dropout=0.0,
                 use_time_attn=False, use_spatial_attn=False,
                 spatial_attn_reduction=4,
                 use_layer_norm=False, use_xavier_init=True):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = [int(width) for width in hidden_dim]
        self.clip_mean = float(clip_mean)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)
        self.use_xavier_init = bool(use_xavier_init)
        self.encoder = None
        if seq_len is not None:
            self.encoder = TemporalConvGRUEncoder(
                seq_len=seq_len, raw_obs_dim=raw_obs_dim,
                gru_hidden_dim=gru_hidden_dim, dropout=dropout,
                cnn_channels=cnn_channels, cnn_kernel_size=cnn_kernel_size,
                use_time_attn=use_time_attn,
                use_spatial_attn=use_spatial_attn,
                spatial_attn_reduction=spatial_attn_reduction,
            )
        input_dim = self.encoder.joint_dim if self.encoder is not None else self.state_dim
        layers = []
        for width in self.hidden_dim:
            layers.append(nn.Linear(input_dim, width))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.extend([nn.ReLU(), nn.Dropout(self.dropout)])
            input_dim = width
        self.trunk = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, self.action_dim)
        self.action_clamp = nn.Hardtanh(-self.clip_mean, self.clip_mean)             if self.clip_mean > 0 else nn.Identity()
        self.register_buffer("action_scale", torch.as_tensor(
            (action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(
            (action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        if self.use_xavier_init:
            self.apply(_xavier_init)

    def forward(self, states):
        if self.encoder is not None:
            states = self.encoder(states)
        output = self.action_clamp(self.action_head(self.trunk(states)))
        return torch.tanh(output) * self.action_scale + self.action_bias

    def mean_action(self, states):
        return self(states)

    def mean_action_from_hidden(self, state, hidden):
        """由当前窗口的编码特征解码动作，供冻结基座的残差策略使用。"""
        if self.encoder is None:
            raise ValueError("当前策略没有时序 encoder")
        state = torch.as_tensor(state, dtype=hidden.dtype, device=hidden.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        latent = self.trunk(torch.cat([hidden, state], dim=1))
        output = self.action_clamp(self.action_head(latent))
        return torch.tanh(output) * self.action_scale + self.action_bias

    def export_cfg(self):
        encoder = self.encoder
        scale = self.action_scale.detach().cpu().flatten()
        bias = self.action_bias.detach().cpu().flatten()
        return {
            "policy_class": self.policy_class,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "seq_len": None if encoder is None else encoder.seq_len,
            "raw_obs_dim": self.state_dim if encoder is None else encoder.raw_obs_dim,
            "gru_hidden_dim": None if encoder is None else encoder.gru_hidden_dim,
            "cnn_channels": 0 if encoder is None else encoder.cnn_channels,
            "cnn_kernel_size": 3 if encoder is None else encoder.cnn_kernel_size,
            "use_time_attn": False if encoder is None else encoder.use_time_attn,
            "use_spatial_attn": False if encoder is None else encoder.use_spatial_attn,
            "spatial_attn_reduction": (
                4 if encoder is None else encoder.spatial_attn_reduction
            ),
            "dropout": self.dropout,
            "use_layer_norm": self.use_layer_norm,
            "use_xavier_init": self.use_xavier_init,
            "clip_mean": self.clip_mean,
            "bc_loss_mode": "mse",
            "action_low": (bias - scale).tolist(),
            "action_high": (bias + scale).tolist(),
        }

    @classmethod
    def from_export_cfg(cls, cfg, device="cpu"):
        low = np.asarray(cfg["action_low"], dtype=np.float32)
        high = np.asarray(cfg["action_high"], dtype=np.float32)
        seq_len = cfg.get("seq_len")
        raw_obs_dim = int(cfg["raw_obs_dim"])
        policy = cls(
            state_dim=int(cfg.get("state_dim", int(seq_len or 1) * raw_obs_dim)),
            hidden_dim=cfg["hidden_dim"], action_dim=int(cfg["action_dim"]),
            action_space=Box(low=low, high=high, dtype=np.float32),
            clip_mean=float(cfg.get("clip_mean", 2.0)),
            seq_len=seq_len, raw_obs_dim=raw_obs_dim,
            gru_hidden_dim=int(cfg.get("gru_hidden_dim") or 64),
            cnn_channels=int(cfg.get("cnn_channels", 64)),
            cnn_kernel_size=int(cfg.get("cnn_kernel_size", 3)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_time_attn=bool(cfg.get("use_time_attn", False)),
            use_spatial_attn=bool(cfg.get("use_spatial_attn", False)),
            spatial_attn_reduction=int(cfg.get("spatial_attn_reduction", 4)),
            use_layer_norm=bool(cfg.get("use_layer_norm", False)),
            use_xavier_init=bool(cfg.get("use_xavier_init", True)),
        )
        return policy.to(device)


def load_base_policy(cfg, device="cpu"):
    """按配置重建普通 BC 基座。"""
    if cfg.get("policy_class", "BasePolicy") != "BasePolicy":
        raise ValueError(f"不支持的基座策略: {cfg['policy_class']}")
    return BasePolicy.from_export_cfg(cfg, device)