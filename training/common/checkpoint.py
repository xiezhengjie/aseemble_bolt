"""权重文件：统一 ``.pt``；加载时兼容旧的 ``.pth``。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

EXT = ".pt"
_LEGACY = ".pth"
POLICY_CFG_NAME = "policy_cfg.json"


# 权重文件路径相关
def _candidates(model_dir: Path, stem: str) -> list[Path]:
    """返回所有候选权重文件路径。"""
    d = Path(model_dir)
    return [d / f"{stem}{EXT}", d / f"{stem}{_LEGACY}"]


def weight_file(model_dir: Path | str, stem: str) -> Path:
    """返回权重文件路径。"""
    d = Path(model_dir)
    return d / f"{stem}{EXT}"


def find_weight(model_dir: Path | str, stem: str) -> Path:
    """查找权重文件路径。"""
    for p in _candidates(Path(model_dir), stem):
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"找不到 {stem}{EXT}（或旧版 {stem}{_LEGACY}）：{model_dir}"
    )


def has_weight(model_dir: Path | str, stem: str) -> bool:
    """检查是否存在权重文件。"""
    return any(p.is_file() for p in _candidates(Path(model_dir), stem))


def load_state_dict(model_dir: Path | str, stem: str, map_location=None):
    """加载权重文件。"""
    path = find_weight(model_dir, stem)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def save_state_dict(state, model_dir: Path | str, stem: str) -> Path:
    """保存权重文件。"""
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}{EXT}"
    torch.save(state, path)
    return path

# 策略配置文件路径相关
def save_policy_cfg(cfg: dict, model_dir: Path | str) -> Path:
    """保存策略配置文件。"""
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / POLICY_CFG_NAME
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path

def load_policy_cfg(model_dir: Path | str) -> dict | None:
    """加载策略配置文件。"""
    path = Path(model_dir) / POLICY_CFG_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
