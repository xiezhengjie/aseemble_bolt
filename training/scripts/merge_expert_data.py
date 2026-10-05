"""合并普通示教数据和干预示教数据。

两个输入文件都必须是按时间顺序排列的完整 episode。支持普通示教文件的
``dones`` 字段和干预文件当前使用的 ``done`` 字段，输出统一为：

``states``
    ``(N, obs_dim)`` 的单帧观测。
``actions``
    ``(N, action_dim)`` 的环境实际动作。
``dones``
    ``(N,)`` 的 episode 终止标记。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import rl_utils


def _load_source(path: str | Path, source_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取并校验一个专家数据文件，统一返回 ``states/actions/dones``。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{source_name} 文件不存在: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"states", "actions"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{source_name} 缺少字段 {missing}: {path}")

        done_keys = [key for key in ("dones", "done") if key in data.files]
        if not done_keys:
            raise ValueError(f"{source_name} 缺少 `done` 或 `dones` 字段: {path}")
        if len(done_keys) > 1:
            raise ValueError(f"{source_name} 同时存在 `done` 和 `dones`，无法确定边界: {path}")

        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        dones = np.asarray(data[done_keys[0]], dtype=np.float32).reshape(-1)

    if states.ndim != 2:
        raise ValueError(f"{source_name}.states 必须是二维数组，实际为 {states.shape}")
    if actions.ndim != 2:
        raise ValueError(f"{source_name}.actions 必须是二维数组，实际为 {actions.shape}")
    if len(states) != len(actions) or len(states) != len(dones):
        raise ValueError(
            f"{source_name} 长度不一致: states={len(states)}, "
            f"actions={len(actions)}, dones={len(dones)}"
        )
    if len(states) == 0:
        raise ValueError(f"{source_name} 为空，不能作为专家数据源: {path}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"{source_name} 包含 NaN 或 Inf: {path}")
    if not np.isfinite(dones).all() or np.any((dones < -1e-6) | (dones > 1.0 + 1e-6)):
        raise ValueError(f"{source_name}.dones 必须是 0/1 标记: {path}")
    dones = (dones > 0.5).astype(np.float32)
    if dones[-1] < 0.5:
        raise ValueError(
            f"{source_name} 最后一帧 done=0，说明最后一个 episode 不完整，拒绝合并: {path}"
        )

    return states, actions, dones


def merge_expert_data(
    demonstration_path: str | Path,
    intervention_path: str | Path,
    output_path: str | Path,
) -> dict[str, int | tuple[int, ...]]:
    """合并普通示教和干预示教，返回合并统计信息。

    两个源文件各自先完成完整性校验，再沿 transition 维度拼接；因此第一个
    文件的最后一个 ``done=1`` 与第二个文件的第一帧之间不会被视为同一条轨迹。
    """
    demo_states, demo_actions, demo_dones = _load_source(
        demonstration_path, "demonstration"
    )
    intervention_states, intervention_actions, intervention_dones = _load_source(
        intervention_path, "intervention"
    )

    if demo_states.shape[1:] != intervention_states.shape[1:]:
        raise ValueError(
            "states 维度不一致: "
            f"demonstration={demo_states.shape}, intervention={intervention_states.shape}"
        )
    if demo_actions.shape[1:] != intervention_actions.shape[1:]:
        raise ValueError(
            "actions 维度不一致: "
            f"demonstration={demo_actions.shape}, intervention={intervention_actions.shape}"
        )

    states = np.concatenate([demo_states, intervention_states], axis=0)
    actions = np.concatenate([demo_actions, intervention_actions], axis=0)
    dones = np.concatenate([demo_dones, intervention_dones], axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez_compressed(tmp_path, states=states, actions=actions, dones=dones)
    os.replace(tmp_path, output_path)

    stats = {
        "demonstration_episodes": int(np.sum(demo_dones > 0.5)),
        "demonstration_transitions": int(len(demo_states)),
        "intervention_episodes": int(np.sum(intervention_dones > 0.5)),
        "intervention_transitions": int(len(intervention_states)),
        "episodes": int(np.sum(dones > 0.5)),
        "transitions": int(len(states)),
        "state_shape": tuple(states.shape),
        "action_shape": tuple(actions.shape),
    }
    print(
        f"专家数据已合并：{output_path}；"
        f"demonstration={stats['demonstration_episodes']} episodes/"
        f"{stats['demonstration_transitions']} transitions；"
        f"intervention={stats['intervention_episodes']} episodes/"
        f"{stats['intervention_transitions']} transitions；"
        f"total={stats['episodes']} episodes/{stats['transitions']} transitions"
    )
    return stats


def parse_args() -> argparse.Namespace:
    root = rl_utils.find_project_root()
    parser = argparse.ArgumentParser(description="合并普通示教和干预专家数据")
    parser.add_argument(
        "--demonstration",
        type=Path,
        default=root / "datasets" / "recorded_expert_data.npz",
    )
    parser.add_argument(
        "--intervention",
        type=Path,
        default=root / "datasets" / "intervention_data_bc_dagger.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets" / "expert_data_combined.npz",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_expert_data(args.demonstration, args.intervention, args.output)
