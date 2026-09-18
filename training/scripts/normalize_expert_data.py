"""将未归一化的专家 npz 转为物理归一化版本（文件名加 _norm 后缀）。

公式与 AssembleMuJoCoEnv 一致：diag(W)⁻¹(p−p_g)、diag(F_max)⁻¹F。
默认尺度：W=[0.04, 0.04, 0.16, 30°]，F_max=[50,50,50,5,5,5]
默认: recorded_data.npz → recorded_data_norm.npz

用法:
  python scripts/normalize_expert_data.py
  python scripts/normalize_expert_data.py --input datasets/bat_7/recorded_data.npz
  python scripts/normalize_expert_data.py --all-bats
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rl_utils import find_project_root

RAW_OBS_DIM = 10
# 对齐 AssembleMuJoCoEnv 默认 workspace / YAW_MAX / 力力矩终止阈值
DEFAULT_W = np.array([0.04, 0.04, 0.16, np.deg2rad(30.0)], dtype=np.float64)
DEFAULT_F_MAX = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0], dtype=np.float64)
DEFAULT_SCALE = np.concatenate([DEFAULT_W, DEFAULT_F_MAX]).astype(np.float32)


def out_path_for(src: Path) -> Path:
    """foo.npz → foo_norm.npz；已带 _norm 则原样返回。"""
    if src.stem.endswith("_norm"):
        return src
    return src.with_name(f"{src.stem}_norm{src.suffix}")


def convert_one(src: Path, dst: Path = None, force: bool = False) -> Path:
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"找不到专家数据: {src}")
    dst = Path(dst) if dst is not None else out_path_for(src)
    if dst.is_file() and not force:
        print(f"[skip] 已存在 {dst}（加 --force 覆盖）")
        return dst

    data = np.load(src, allow_pickle=True)
    if "states" not in data.files or "actions" not in data.files:
        raise KeyError(f"{src} 缺少 states/actions，files={list(data.files)}")

    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    dones = np.asarray(data["dones"], dtype=np.float32) if "dones" in data.files else None

    if states.ndim != 2 or states.shape[1] != RAW_OBS_DIM:
        print(f"[skip] {src} shape={states.shape}（需要 (N, {RAW_OBS_DIM})）")
        return None

    states_n = (states / DEFAULT_SCALE).astype(np.float32)
    payload = {"states": states_n, "actions": actions}
    if dones is not None:
        payload["dones"] = dones
    payload["W"] = DEFAULT_W.astype(np.float32)
    payload["F_max"] = DEFAULT_F_MAX.astype(np.float32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)
    abs_max = np.max(np.abs(states_n), axis=0)
    print(f"[ok] {src.name} → {dst}  N={len(states_n)}  "
          f"|s|_max={np.array2string(abs_max, precision=3)}")
    return dst


def main():
    root = find_project_root()
    default_in = root / "datasets" / "recorded_data.npz"
    p = argparse.ArgumentParser(description="专家数据物理归一化（加 _norm 后缀）")
    p.add_argument("--input", type=str, default=str(default_in),
                   help="原始 npz 路径")
    p.add_argument("--output", type=str, default="",
                   help="输出路径；默认在 stem 后加 _norm")
    p.add_argument("--all-bats", action="store_true",
                   help="同时转换 datasets/bat_*/recorded_data.npz")
    p.add_argument("--force", action="store_true", help="覆盖已存在的 *_norm.npz")
    args = p.parse_args()

    convert_one(Path(args.input),
                Path(args.output) if args.output else None,
                force=args.force)

    if args.all_bats:
        for bat in sorted((root / "datasets").glob("bat_*/recorded_data.npz")):
            convert_one(bat, force=args.force)


if __name__ == "__main__":
    main()
