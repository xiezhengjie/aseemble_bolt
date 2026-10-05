"""运行 BC 基座并将闭环轨迹与专家数据做近邻比较。

示例：
    /usr/bin/python3.10 training/tests/bc_failure_diagnosis.py --episodes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.model.base.basic_model import load_base_policy
from training.envs.assemble_mujoco_env import AssembleMuJoCoEnv
from training.common import rl_utils
from training.common.checkpoint import load_policy_cfg, load_state_dict
from training.common.rl_utils import RunningMeanStd


def phase_proxy(raw_state: np.ndarray) -> int:
    """用保存的10维观测近似 R/S/I 阶段；真实 rollout 阶段使用 info['state']。"""
    scale = np.array([.04, .04, .009, np.deg2rad(30), 50, 50, 50, 5, 5, 5])
    x = np.asarray(raw_state, dtype=np.float64) * scale
    xy = float(np.linalg.norm(x[:2]))
    depth = float(-x[2])
    yaw = abs(float(x[3]))
    if xy > .008 or depth < -.004:
        return 0
    if xy > .003 or yaw > np.deg2rad(7) or depth < -.003:
        return 1
    return 2


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=100000)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--model-dir", type=Path, default=root / "models" / "bc_model_ur5e")
    ap.add_argument("--expert", type=Path, default=root / "datasets" / "recorded_expert_data.npz")
    ap.add_argument("--output", type=Path, default=root / "logs" / "bc_failure_diagnosis.npz")
    args = ap.parse_args()

    cfg = load_policy_cfg(args.model_dir)
    if not cfg:
        raise RuntimeError(f"缺少 policy_cfg.json: {args.model_dir}")
    frame_stack = int(cfg.get("seq_len") or 1)
    env0 = AssembleMuJoCoEnv(
        xml_path=str(root / "assets" / "mjcf" / "ur5e_assemble_sence.xml),
        urdf_path=str(root / "assets" / "urdf" / "ur5e_assemble.urdf"),
        render_mode=None,
        max_episodic_steps=args.max_steps,
    )
    env = rl_utils.wrap_frame_stack(env0, frame_stack, padding_type="reset")
    policy = load_base_policy(cfg, "cpu")
    policy.load_state_dict(load_state_dict(args.model_dir, "policy_net", map_location="cpu"))
    policy.eval()
    obs_normalizer = None
    normalizer_path = args.model_dir / "obs_normalizer.npz"
    if bool(cfg.get("obs_normalized", False)) and normalizer_path.exists():
        obs_normalizer = RunningMeanStd(shape=(int(cfg.get("raw_obs_dim") or 10),))
        obs_normalizer.load_normalizer(args.model_dir)
    expert_data = np.load(args.expert, allow_pickle=True)
    expert_states = np.asarray(expert_data["states"], dtype=np.float32)
    expert_actions = np.asarray(expert_data["actions"], dtype=np.float32)
    tree = cKDTree(expert_states.astype(np.float64))

    records = []
    episodes = []
    for ep in range(args.episodes):
        obs, reset_info = env.reset(seed=args.seed_offset + ep)
        done = False
        rows = []
        while not done:
            raw_state = np.asarray(env0._get_observation(), dtype=np.float32).copy()
            with torch.no_grad():
                policy_obs = (
                    obs_normalizer.normalize(np.asarray(obs, dtype=np.float32))
                    if obs_normalizer is not None
                    else np.asarray(obs, dtype=np.float32)
                )
                obs_t = torch.as_tensor(policy_obs[None], dtype=torch.float32)
                action = policy.mean_action(obs_t)[0].cpu().numpy().astype(np.float32)
            before_state = int(env0.current_state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            expert_dist, expert_idx = tree.query(raw_state.astype(np.float64), k=1)
            expert_action = expert_actions[int(expert_idx)]
            rows.append({
                "episode": ep,
                "step": len(rows),
                "state": raw_state,
                "action": action.copy(),
                "expert_action_nn": expert_action.copy(),
                "state_nn_dist": float(expert_dist),
                "action_mse_nn": float(np.mean((action - expert_action) ** 2)),
                "action_mae_nn": float(np.mean(np.abs(action - expert_action))),
                "phase_before": before_state,
                "phase_after": int(info.get("state", -1)),
                "reward": float(reward),
                "depth": float(info.get("depth", np.nan)),
                "position_error_xy": float(info.get("position_error_xy", np.nan)),
                "yaw_error": float(info.get("yaw_error", np.nan)),
                "force": np.asarray(info.get("force", [np.nan] * 3), dtype=np.float32),
                "torque": np.asarray(info.get("torque", [np.nan] * 3), dtype=np.float32),
                "success": bool(info.get("success", False)),
                "fail_contact": bool(info.get("fail_contact", False)),
                "fail_workspace": bool(info.get("fail_workspace", False)),
            })
            obs = next_obs
        final = rows[-1] if rows else {}
        episodes.append({
            "episode": ep,
            "seed": args.seed_offset + ep,
            "length": len(rows),
            "success": bool(final.get("success", False)),
            "fail_contact": bool(final.get("fail_contact", False)),
            "fail_workspace": bool(final.get("fail_workspace", False)),
            "final_state": int(final.get("phase_after", -1)),
            "final_depth": float(final.get("depth", np.nan)),
            "final_position_error_xy": float(final.get("position_error_xy", np.nan)),
            "final_yaw_error": float(final.get("yaw_error", np.nan)),
        })
        records.extend(rows)
        print(
            f"episode={ep} len={len(rows)} success={episodes[-1]['success']} "
            f"final_state={episodes[-1]['final_state']} "
            f"depth={episodes[-1]['final_depth']:.5f} "
            f"xy={episodes[-1]['final_position_error_xy']:.5f} "
            f"yaw={episodes[-1]['final_yaw_error']:.3f}",
            flush=True,
        )

    # 汇总阶段和误差，找出第一个明显偏离专家动作/状态的步骤。
    by_phase = defaultdict(list)
    for r in records:
        by_phase[int(r["phase_before"])].append(r)
    summary = {}
    for phase, rs in sorted(by_phase.items()):
        summary[str(phase)] = {
            "n": len(rs),
            "state_nn_dist_mean": float(np.mean([r["state_nn_dist"] for r in rs])),
            "state_nn_dist_p95": float(np.percentile([r["state_nn_dist"] for r in rs], 95)),
            "action_mse_nn_mean": float(np.mean([r["action_mse_nn"] for r in rs])),
            "action_mae_nn_mean": float(np.mean([r["action_mae_nn"] for r in rs])),
            "max_action_mse_step": int(max(rs, key=lambda r: r["action_mse_nn"])["step"]),
        }
    for ep in range(args.episodes):
        rs = [r for r in records if r["episode"] == ep]
        bad = [r for r in rs if r["state_nn_dist"] > 0.15 or r["action_mse_nn"] > 0.05]
        if bad:
            first = min(bad, key=lambda r: r["step"])
            episodes[ep]["first_large_deviation"] = {
                "step": int(first["step"]),
                "phase": int(first["phase_before"]),
                "state_nn_dist": float(first["state_nn_dist"]),
                "action_mse_nn": float(first["action_mse_nn"]),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # npz 保存逐步数组，json 保存结构化汇总。
    np.savez_compressed(
        args.output,
        states=np.stack([r["state"] for r in records]),
        actions=np.stack([r["action"] for r in records]),
        expert_actions_nn=np.stack([r["expert_action_nn"] for r in records]),
        state_nn_dist=np.asarray([r["state_nn_dist"] for r in records], dtype=np.float32),
        action_mse_nn=np.asarray([r["action_mse_nn"] for r in records], dtype=np.float32),
        episode=np.asarray([r["episode"] for r in records], dtype=np.int32),
        step=np.asarray([r["step"] for r in records], dtype=np.int32),
        phase=np.asarray([r["phase_before"] for r in records], dtype=np.int32),
        depth=np.asarray([r["depth"] for r in records], dtype=np.float32),
        position_error_xy=np.asarray([r["position_error_xy"] for r in records], dtype=np.float32),
        yaw_error=np.asarray([r["yaw_error"] for r in records], dtype=np.float32),
    )
    report = {
        "model_dir": str(args.model_dir),
        "expert": str(args.expert),
        "frame_stack": frame_stack,
        "obs_normalized": obs_normalizer is not None,
        "episodes": episodes,
        "summary_by_phase": summary,
        "success_rate": float(np.mean([e["success"] for e in episodes])) if episodes else 0.0,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}\nreport: {report_path}")
    env0.close()


if __name__ == "__main__":
    main()
