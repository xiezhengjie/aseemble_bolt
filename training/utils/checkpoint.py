"""权重文件：统一 ``.pt``；加载时兼容旧的 ``.pth``。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

EXT = ".pt"
_LEGACY = ".pth"
POLICY_CFG_NAME = "policy_cfg.json"


def _candidates(model_dir: Path, stem: str) -> list[Path]:
    d = Path(model_dir)
    return [d / f"{stem}{EXT}", d / f"{stem}{_LEGACY}"]


def weight_file(model_dir: Path | str, stem: str) -> Path:
    return Path(model_dir) / f"{stem}{EXT}"


def find_weight(model_dir: Path | str, stem: str) -> Path:
    for p in _candidates(Path(model_dir), stem):
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"找不到 {stem}{EXT}（或旧版 {stem}{_LEGACY}）：{model_dir}"
    )


def has_weight(model_dir: Path | str, stem: str) -> bool:
    return any(p.is_file() for p in _candidates(Path(model_dir), stem))


def load_state_dict(model_dir: Path | str, stem: str, map_location=None):
    path = find_weight(model_dir, stem)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_state_dict(state, model_dir: Path | str, stem: str) -> Path:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}{EXT}"
    torch.save(state, path)
    return path


def save_policy_cfg(cfg: dict, model_dir: Path | str) -> Path:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / POLICY_CFG_NAME
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def load_policy_cfg(model_dir: Path | str) -> dict | None:
    path = Path(model_dir) / POLICY_CFG_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cfg_from_state_dict(state: dict, *, seq_len: int = 8) -> dict:
    """没有 ``policy_cfg.json`` 时，从 state_dict 反推结构。

    有 GRU 编码器时默认 ``seq_len=8``（权重里读不出 T）；
    旧版单帧 MLP（``fc_latent.0`` 输入为 10）则 ``seq_len=None``。
    """
    hidden: list[int] = []
    i = 0
    while f"fc_latent.{i}.weight" in state:
        hidden.append(int(state[f"fc_latent.{i}.weight"].shape[0]))
        i += 2
    scale = state["action_scale"].detach().cpu().flatten()
    bias = state["action_bias"].detach().cpu().flatten()
    log_std = state.get("log_std")
    has_encoder = "encoder.gru.weight_ih_l0" in state
    if has_encoder:
        w = state["encoder.gru.weight_ih_l0"]
        gru_hidden_dim = int(w.shape[0] // 3)
        raw_obs_dim = int(w.shape[1])
        cfg_seq: int | None = int(seq_len)
        use_h_norm = "encoder.h_norm.weight" in state
    else:
        gru_hidden_dim = None
        raw_obs_dim = int(state["fc_latent.0.weight"].shape[1])
        cfg_seq = None
        use_h_norm = False
    return {
        "hidden_dim": hidden,
        "action_dim": int(state["fc_mu.weight"].shape[0]),
        "seq_len": cfg_seq,
        "raw_obs_dim": raw_obs_dim,
        "gru_hidden_dim": gru_hidden_dim,
        "use_h_norm": use_h_norm,
        "clip_mean": 2.0,
        "log_std_init": -3.0,
        "use_sde": bool(log_std is not None and getattr(log_std, "ndim", 1) == 2),
        "action_low": (bias - scale).tolist(),
        "action_high": (bias + scale).tolist(),
    }
