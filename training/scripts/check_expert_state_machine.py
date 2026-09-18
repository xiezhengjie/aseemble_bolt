"""离线回放 _state_trans，检验专家轨迹能走到哪个状态。

专家观测:
  10 维: [dx, dy, dz, yaw, fx, fy, fz, tx, ty, tz]
   9 维: [dx, dy, dz, fx, fy, fz, tx, ty, tz]  (无 yaw，S→I 的偏航门限按 0 处理)

约定: depth = -pos_z （与环境 _get_assemble_depth 一致）。
缺失量:
  - angle_z 观测中没有，默认视为 0（乐观，仅检验其余坐底条件）
  - max_calibrated_ft 用当前步 ft 代替（专家每步只有一个力样本）
  - 工作空间用 relative_pos 近似 eef - base_target（忽略目标随机偏移 ~1–3mm）
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np

from utils.rl_utils import find_project_root


STATE_NAME = {0: "R接近", 1: "S搜索", 2: "I插入", 3: "T成功", 4: "F失败"}


class OfflineStateMachine:
    """与 AssembleMuJoCoEnv._state_trans 对齐的离线状态机。"""

    def __init__(
        self,
        assembly_depth_threshold=0.0045,
        success_pos_threshold=0.004,
        success_reach_hole_threshold=0.01044,
        success_force_threshold=1.5,
        success_torque_threshold=0.05,
        force_threshold_terminate=50.0,
        torque_threshold_terminate=5.0,
        workspace=None,
        skip_angle_z=True,
    ):
        self.assembly_depth_threshold = assembly_depth_threshold
        self.success_pos_threshold = success_pos_threshold
        self.success_reach_hole_threshold = success_reach_hole_threshold
        self.success_force_threshold = success_force_threshold
        self.success_torque_threshold = success_torque_threshold
        self.force_threshold_terminate = force_threshold_terminate
        self.torque_threshold_terminate = torque_threshold_terminate
        self.workspace = np.array([[-0.04, -0.04, -0.012],
                                   [0.04, 0.04, 0.12]] if workspace is None else workspace)
        self.skip_angle_z = skip_angle_z
        self.reset()

    def reset(self, depth0=0.0):
        self.current_state = 0
        self.seat_count = 0
        self.backwards_count = 0
        self.prev_depth = depth0

    def step(self, eef_pos, depth, pos_error, yaw_error, pos_error_xy, ft, angle_z=0.0):
        f_norm = np.linalg.norm(ft[:3])
        t_norm = np.linalg.norm(ft[3:])
        f_max_norm = f_norm
        t_max_norm = t_norm

        fail_reason = None
        if (f_max_norm > self.force_threshold_terminate or
            t_max_norm > self.torque_threshold_terminate or
            np.any(eef_pos < self.workspace[0]) or
            np.any(eef_pos > self.workspace[1])):
            if f_max_norm > self.force_threshold_terminate or t_max_norm > self.torque_threshold_terminate:
                fail_reason = "force/torque"
            else:
                fail_reason = "workspace"
            self.current_state = 4
            self.prev_depth = depth
            return self.current_state, fail_reason

        if self.current_state == 0:
            if (pos_error_xy <= 0.008 and depth >= -0.004):
                self.current_state = 1
            else:
                if (depth - self.prev_depth) < -0.0005:
                    self.backwards_count += 1
                elif (depth - self.prev_depth) >= 0.0005:
                    self.backwards_count = 0
                if self.backwards_count > 6:
                    self.current_state = 4
                    fail_reason = "backwards"
                    self.prev_depth = depth
                    return self.current_state, fail_reason

        if self.current_state == 1:
            if (pos_error_xy <= 0.003 and
                np.abs(yaw_error) <= np.deg2rad(7) and
                depth >= -0.003):
                self.current_state = 2

        if self.current_state == 2:
            if (depth >= self.assembly_depth_threshold and
                abs(depth - self.prev_depth) < 2e-4 and
                pos_error_xy <= 0.0015 and
                np.abs(ft[2]) >= self.success_force_threshold and
                np.linalg.norm(ft[:2]) <= 0.3 * self.success_force_threshold and
                np.linalg.norm(ft[3:]) <= self.success_torque_threshold):
                self.seat_count += 1
            else:
                self.seat_count = 0
            if self.seat_count >= 3:
                self.current_state = 3

        self.prev_depth = depth
        return self.current_state, fail_reason

    def seat_flags(self, depth, prev_depth, pos_error_xy, ft, angle_z=0.0):
        """单帧坐底 7 条件（不含连续 3 拍）。"""
        angle_ok = True if self.skip_angle_z else (np.abs(angle_z) <= 1)
        flags = {
            "depth>=th": depth >= self.assembly_depth_threshold,
            "depth_stagnant": abs(depth - prev_depth) < 2e-4,
            "xy<=1.5mm": pos_error_xy <= 0.0015,
            "|fz|>=th": np.abs(ft[2]) >= self.success_force_threshold,
            "|fxy|<=0.3th": np.linalg.norm(ft[:2]) <= 0.3 * self.success_force_threshold,
            "|t|<=th": np.linalg.norm(ft[3:]) <= self.success_torque_threshold,
        }
        flags["all"] = all(flags.values())
        return flags


def parse_obs(obs):
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape[-1] == 10:
        pos, yaw, ft = obs[:3], float(obs[3]), obs[4:10]
        has_yaw = True
    elif obs.shape[-1] == 9:
        pos, yaw, ft = obs[:3], 0.0, obs[3:9]
        has_yaw = False
    else:
        raise ValueError(f"unexpected obs dim {obs.shape}")
    depth = -pos[2]  # pos_z 与 depth 符号相反
    pos_error = float(np.linalg.norm(pos))
    pos_error_xy = float(np.linalg.norm(pos[:2]))
    return {
        "eef_pos": pos.copy(),
        "depth": float(depth),
        "pos_error": pos_error,
        "yaw_error": yaw,
        "pos_error_xy": pos_error_xy,
        "ft": ft.copy(),
        "has_yaw": has_yaw,
        "pos_z": float(pos[2]),
    }


def split_episodes(states, dones):
    states = np.asarray(states)
    dones = np.asarray(dones)
    ends = np.where(dones > 0.5)[0]
    if len(ends) == 0:
        return [(0, len(states))]
    starts = np.concatenate([[0], ends[:-1] + 1])
    eps = list(zip(starts.tolist(), (ends + 1).tolist()))
    if ends[-1] < len(states) - 1:
        # 末尾未完成片段丢弃
        pass
    return eps


def replay_episode(states, sm: OfflineStateMachine, pad_last=False):
    states = np.asarray(states)
    if pad_last and len(states) > 0:
        # 录制存的是 step 前观测，成功当拍的 post-step 未写入；复制末帧补这一拍
        states = np.vstack([states, states[-1:]])
    parsed0 = parse_obs(states[0])
    sm.reset(depth0=parsed0["depth"])
    hist = []
    first_enter = {0: 0, 1: None, 2: None, 3: None, 4: None}
    fail_reason = None
    max_seat = 0

    for t, obs in enumerate(states):
        p = parse_obs(obs)
        prev_d = sm.prev_depth
        st, reason = sm.step(
            p["eef_pos"], p["depth"], p["pos_error"], p["yaw_error"],
            p["pos_error_xy"], p["ft"],
        )
        if first_enter.get(st) is None:
            first_enter[st] = t
        max_seat = max(max_seat, sm.seat_count)
        flags = sm.seat_flags(p["depth"], prev_d, p["pos_error_xy"], p["ft"])
        hist.append({
            "t": t, "state": st, "reason": reason,
            "seat_count": sm.seat_count, **p, "flags": flags, "prev_depth": prev_d,
        })
        if reason:
            fail_reason = reason
        if st in (3, 4):
            break
    return {
        "final_state": hist[-1]["state"],
        "first_enter": first_enter,
        "fail_reason": fail_reason,
        "max_seat": max_seat,
        "hist": hist,
        "has_yaw": parsed0["has_yaw"],
        "T": len(states),
        "used": len(hist),
    }


def best_seat_frame(hist):
    """选坐底条件命中数最多、其次 depth 最大的帧。"""
    def key(h):
        n = sum(1 for k, v in h["flags"].items() if k != "all" and v)
        return (n, h["depth"], -h["t"])
    return max(hist, key=key)


def fmt_mm(x):
    return f"{x * 1000:.3f}mm"


def fmt_deg(rad):
    return f"{np.degrees(rad):.2f}deg"


def analyze_file(path, sm_kwargs=None, pad_last=False):
    data = np.load(path, allow_pickle=True)
    states, dones = data["states"], data["dones"]
    eps = split_episodes(states, dones)
    sm = OfflineStateMachine(**(sm_kwargs or {}))

    results = []
    for s, e in eps:
        if e - s < 2:
            continue
        results.append(replay_episode(states[s:e], sm, pad_last=pad_last))

    n = len(results)
    if n == 0:
        return {"path": str(path), "n": 0}

    counts = {k: 0 for k in range(5)}
    fail_reasons = {}
    for r in results:
        counts[r["final_state"]] += 1
        if r["fail_reason"]:
            fail_reasons[r["fail_reason"]] = fail_reasons.get(r["fail_reason"], 0) + 1

    # 转移到达率
    reached = {k: sum(1 for r in results if r["first_enter"][k] is not None or r["final_state"] >= k)
               for k in range(4)}
    # 更干净：用 first_enter / final
    reached = {k: 0 for k in range(5)}
    for r in results:
        for k in range(5):
            if r["first_enter"].get(k) is not None or r["final_state"] == k:
                if k <= r["final_state"] or (k == 4 and r["final_state"] == 4):
                    pass
        # 路径上出现过的状态
        seen = set(h["state"] for h in r["hist"])
        for k in seen:
            reached[k] += 1

    enter_step = {k: [] for k in range(1, 4)}
    for r in results:
        for k in range(1, 4):
            t = r["first_enter"].get(k)
            if t is not None:
                enter_step[k].append(t)

    stuck = {0: [], 1: [], 2: []}
    for r in results:
        if r["final_state"] in stuck:
            stuck[r["final_state"]].append(r)

    def bottleneck_R(rs):
        """卡在接近: 看全轨迹是否曾满足 R→S，以及 backwards/pos_error/depth。"""
        if not rs:
            return {"n": 0}
        never_pos, never_depth, both_at_some = 0, 0, 0
        min_pos, max_depth = [], []
        for r in rs:
            pos_ok = any(h["pos_error_xy"] <= 0.008 for h in r["hist"])
            depth_ok = any(h["depth"] >= -0.004 for h in r["hist"])
            both = any(
                h["pos_error_xy"] <= 0.008 and h["depth"] >= -0.004
                for h in r["hist"]
            )
            if not pos_ok:
                never_pos += 1
            if not depth_ok:
                never_depth += 1
            if both:
                both_at_some += 1
            min_pos.append(min(h["pos_error_xy"] for h in r["hist"]))
            max_depth.append(max(h["depth"] for h in r["hist"]))
        return {
            "n": len(rs),
            "never_pos_ok": never_pos,
            "never_depth_ok": never_depth,
            "both_ok_somewhere": both_at_some,
            "min_pos_error_mm": (np.min(min_pos) * 1000, np.median(min_pos) * 1000, np.max(min_pos) * 1000),
            "max_depth_mm": (np.min(max_depth) * 1000, np.median(max_depth) * 1000, np.max(max_depth) * 1000),
        }

    def bottleneck_S(rs):
        if not rs:
            return {"n": 0}
        n_xy = n_yaw = n_depth = n_all = 0
        mins = {"xy": [], "yaw": [], "depth": []}
        for r in rs:
            xy_ok = any(h["pos_error_xy"] <= 0.003 for h in r["hist"])
            yaw_ok = any(abs(h["yaw_error"]) <= np.deg2rad(7) for h in r["hist"])
            d_ok = any(h["depth"] >= -0.003 for h in r["hist"])
            all_ok = any(
                h["pos_error_xy"] <= 0.003 and abs(h["yaw_error"]) <= np.deg2rad(7) and h["depth"] >= -0.003
                for h in r["hist"]
            )
            n_xy += int(xy_ok)
            n_yaw += int(yaw_ok)
            n_depth += int(d_ok)
            n_all += int(all_ok)
            mins["xy"].append(min(h["pos_error_xy"] for h in r["hist"]))
            mins["yaw"].append(min(abs(h["yaw_error"]) for h in r["hist"]))
            mins["depth"].append(max(h["depth"] for h in r["hist"]))
        return {
            "n": len(rs),
            "ever_xy_ok": n_xy,
            "ever_yaw_ok": n_yaw,
            "ever_depth_ok": n_depth,
            "ever_all_ok": n_all,
            "best_xy_mm": tuple(np.percentile(mins["xy"], [0, 50, 100]) * 1000) if rs else None,
            "best_yaw_deg": tuple(np.degrees(np.percentile(mins["yaw"], [0, 50, 100]))) if rs else None,
            "best_depth_mm": tuple(np.percentile(mins["depth"], [0, 50, 100]) * 1000) if rs else None,
        }

    def bottleneck_I(rs):
        keys = ["depth>=th", "depth_stagnant", "xy<=1.5mm", "|fz|>=th",
                "|fxy|<=0.3th", "|t|<=th", "all"]
        if not rs:
            return {"n": 0, "ever": {k: 0 for k in keys}, "at_best": {k: 0 for k in keys},
                    "max_seat": None, "extras": []}
        ever = {k: 0 for k in keys}
        at_best = {k: 0 for k in keys}
        max_seats = []
        extras = []
        for r in rs:
            max_seats.append(r["max_seat"])
            ever_flags = {k: False for k in keys}
            for h in r["hist"]:
                for k in keys:
                    ever_flags[k] = ever_flags[k] or h["flags"][k]
            for k in keys:
                ever[k] += int(ever_flags[k])
            bf = best_seat_frame(r["hist"])
            for k in keys:
                at_best[k] += int(bf["flags"][k])
            extras.append({
                "best_t": bf["t"],
                "depth_mm": bf["depth"] * 1000,
                "ddepth_mm": abs(bf["depth"] - bf["prev_depth"]) * 1000,
                "xy_mm": bf["pos_error_xy"] * 1000,
                "fz": bf["ft"][2],
                "fxy": float(np.linalg.norm(bf["ft"][:2])),
                "tn": float(np.linalg.norm(bf["ft"][3:])),
                "yaw_deg": np.degrees(bf["yaw_error"]),
                "pos_z_mm": bf["pos_z"] * 1000,
                "max_seat": r["max_seat"],
            })
        return {"n": len(rs), "ever": ever, "at_best": at_best,
                "max_seat": (min(max_seats), float(np.median(max_seats)), max(max_seats)) if rs else None,
                "extras": extras}

    # 全局量级（所有 episode 的极值）
    glob = {
        "min_pos_xy_mm": [],
        "max_depth_mm": [],
        "min_pos_z_mm": [],
        "max_|fz|": [],
        "min_|fz|_at_deep": [],
        "max_fxy_at_deep": [],
        "max_tn_at_deep": [],
        "min_|yaw|_deg": [],
    }
    deep_th = 0.0045
    for r in results:
        glob["min_pos_xy_mm"].append(min(h["pos_error_xy"] for h in r["hist"]) * 1000)
        glob["max_depth_mm"].append(max(h["depth"] for h in r["hist"]) * 1000)
        glob["min_pos_z_mm"].append(min(h["pos_z"] for h in r["hist"]) * 1000)
        glob["max_|fz|"].append(max(abs(h["ft"][2]) for h in r["hist"]))
        glob["min_|yaw|_deg"].append(min(abs(h["yaw_error"]) for h in r["hist"]) * 180 / np.pi)
        deep_frames = [h for h in r["hist"] if h["depth"] >= deep_th]
        if deep_frames:
            glob["min_|fz|_at_deep"].append(min(abs(h["ft"][2]) for h in deep_frames))
            glob["max_fxy_at_deep"].append(max(np.linalg.norm(h["ft"][:2]) for h in deep_frames))
            glob["max_tn_at_deep"].append(max(np.linalg.norm(h["ft"][3:]) for h in deep_frames))

    has_yaw = results[0]["has_yaw"]
    return {
        "path": str(path),
        "n": n,
        "dim": int(states.shape[1]),
        "has_yaw": has_yaw,
        "counts": counts,
        "reached": reached,
        "fail_reasons": fail_reasons,
        "enter_step": {k: (float(np.mean(v)), float(np.median(v)), int(np.min(v)), int(np.max(v)))
                       if v else None for k, v in enter_step.items()},
        "stuck_R": bottleneck_R(stuck[0]),
        "stuck_S": bottleneck_S(stuck[1]),
        "stuck_I": bottleneck_I(stuck[2]),
        "success_I": bottleneck_I([r for r in results if r["final_state"] == 3]),
        "glob": {k: (tuple(np.percentile(v, [0, 50, 100])) if v else None) for k, v in glob.items()},
        "n_deep_eps": sum(1 for r in results if max(h["depth"] for h in r["hist"]) >= deep_th),
        "results": results,
    }


def print_report(rep):
    print("\n" + "=" * 88)
    print(f"文件: {rep['path']}")
    if rep["n"] == 0:
        print("  (无 episode)")
        return
    print(f"episode 数: {rep['n']}  观测维: {rep['dim']}  含 yaw: {rep['has_yaw']}")
    print("-" * 88)
    print("终态分布 (每条轨迹停下时的状态):")
    for k in range(5):
        c = rep["counts"][k]
        pct = 100.0 * c / rep["n"]
        print(f"  {k} {STATE_NAME[k]:<6}: {c:4d}  ({pct:5.1f}%)")
    print("轨迹中曾经进入过的状态:")
    for k in range(5):
        c = rep["reached"][k]
        pct = 100.0 * c / rep["n"]
        print(f"  {k} {STATE_NAME[k]:<6}: {c:4d}  ({pct:5.1f}%)")
    if rep["fail_reasons"]:
        print("失败原因:")
        for k, v in rep["fail_reasons"].items():
            print(f"  {k}: {v}")
    print("首次进入步数 (mean / median / min / max):")
    for k in range(1, 4):
        v = rep["enter_step"][k]
        if v is None:
            print(f"  {k} {STATE_NAME[k]}: 从未进入")
        else:
            print(f"  {k} {STATE_NAME[k]}: {v[0]:.1f} / {v[1]:.1f} / {v[2]} / {v[3]}")

    print("-" * 88)
    print("全局极值 (每条轨迹内的 best，再对轨迹做 min/median/max):")
    g = rep["glob"]
    def gline(name, key, unit=""):
        v = g[key]
        if v is None:
            print(f"  {name}: n/a")
        else:
            print(f"  {name}: min={v[0]:.3f}{unit}  med={v[1]:.3f}{unit}  max={v[2]:.3f}{unit}")
    gline("min pos_xy", "min_pos_xy_mm", "mm")
    gline("max depth (= -pos_z)", "max_depth_mm", "mm")
    gline("min pos_z", "min_pos_z_mm", "mm")
    gline("max |fz|", "max_|fz|", "N")
    gline("min |yaw|", "min_|yaw|_deg", "deg")
    print(f"  depth>=4.5mm 的 episode: {rep['n_deep_eps']}/{rep['n']}")
    if g["min_|fz|_at_deep"]:
        gline("depth>=4.5mm 时 min |fz|", "min_|fz|_at_deep", "N")
        gline("depth>=4.5mm 时 max |fxy|", "max_fxy_at_deep", "N")
        gline("depth>=4.5mm 时 max |t|", "max_tn_at_deep", "Nm")

    if rep["stuck_R"]["n"]:
        b = rep["stuck_R"]
        print("-" * 88)
        print(f"卡在 0 接近 的 {b['n']} 条: R→S 需要 xy<=8mm 且 depth>=-4mm")
        print(f"  全程从未 xy 达标: {b['never_pos_ok']}")
        print(f"  全程从未 depth 达标:     {b['never_depth_ok']}")
        print(f"  两条件曾同时成立(但状态机没转，可能 backwards 先触发): {b['both_ok_somewhere']}")
        print(f"  轨迹内 min pos_error: {b['min_pos_error_mm'][0]:.2f}/{b['min_pos_error_mm'][1]:.2f}/{b['min_pos_error_mm'][2]:.2f} mm")
        print(f"  轨迹内 max depth:     {b['max_depth_mm'][0]:.2f}/{b['max_depth_mm'][1]:.2f}/{b['max_depth_mm'][2]:.2f} mm")

    if rep["stuck_S"]["n"]:
        b = rep["stuck_S"]
        print("-" * 88)
        print(f"卡在 1 搜索 的 {b['n']} 条: S→I 需要 xy<=3mm 且 |yaw|<=7deg 且 depth>=-3mm")
        print(f"  曾 xy 达标:    {b['ever_xy_ok']}/{b['n']}")
        print(f"  曾 yaw 达标:   {b['ever_yaw_ok']}/{b['n']}" + ("" if rep["has_yaw"] else "  (9维无yaw, 按0处理, 恒达标)"))
        print(f"  曾 depth 达标: {b['ever_depth_ok']}/{b['n']}")
        print(f"  三条件曾同时成立: {b['ever_all_ok']}/{b['n']}")
        print(f"  best xy mm  min/med/max: {b['best_xy_mm']}")
        print(f"  best |yaw| deg:          {b['best_yaw_deg']}")
        print(f"  best depth mm:           {b['best_depth_mm']}")

    if rep["stuck_I"]["n"]:
        b = rep["stuck_I"]
        print("-" * 88)
        print(f"卡在 2 插入 的 {b['n']} 条: I→T 需连续 3 拍 (xy<=1.5mm, 已去掉 angle_z)")
        print(f"  轨迹内 max seat_count min/med/max: {b['max_seat']}")
        print("  条件在某帧成立过 / 在最佳候选帧成立:")
        for k in ["depth>=th", "depth_stagnant", "xy<=1.5mm", "|fz|>=th",
                  "|fxy|<=0.3th", "|t|<=th", "all"]:
            print(f"    {k:<16} ever {b['ever'][k]:4d}/{b['n']}   best-frame {b['at_best'][k]:4d}/{b['n']}")
        # 打印若干典型卡住样本
        extras = sorted(b["extras"], key=lambda x: -x["depth_mm"])
        print("  深度最大的 8 条卡在 I 的最佳候选帧:")
        print("    ep_rank  t  depth_mm  ddepth_mm  xy_mm   fz(N)   fxy(N)  |t|(Nm)  yaw_deg  seat")
        for i, e in enumerate(extras[:8]):
            print(f"    {i:7d} {e['best_t']:3d} {e['depth_mm']:8.3f} {e['ddepth_mm']:9.3f} "
                  f"{e['xy_mm']:6.3f} {e['fz']:7.2f} {e['fxy']:7.3f} {e['tn']:8.4f} "
                  f"{e['yaw_deg']:7.2f}  {e['max_seat']}")

    if rep["counts"][3]:
        b = rep["success_I"]
        print("-" * 88)
        print(f"走到 3 成功 的 {rep['counts'][3]} 条: 坐底条件 ever / 最佳帧")
        for k in ["depth>=th", "depth_stagnant", "xy<=1.5mm", "|fz|>=th",
                  "|fxy|<=0.3th", "|t|<=th", "all"]:
            print(f"    {k:<16} ever {b['ever'][k]:4d}/{b['n']}   best-frame {b['at_best'][k]:4d}/{b['n']}")


def main():
    root = find_project_root() / "datasets"
    files = [
        root / "recorded_data.npz",
        root / "recorded_data_filtered.npz",
        root / "bat_1" / "recorded_data.npz",
        root / "bat_2" / "recorded_data.npz",
        root / "bat_3" / "recorded_data.npz",
        root / "bat_4" / "recorded_data.npz",
        root / "bat_5" / "recorded_data.npz",
        root / "bat_6" / "recorded_data.npz",
        root / "bat_7" / "recorded_data.npz",
    ]
    files = [f for f in files if f.exists()]

    print("约定: depth = -pos_z")
    print("R→S: xy<=8mm 且 depth>=-4mm")
    print("S→I: xy<=3mm 且 |yaw|<=7deg 且 depth>=-3mm")
    print("I→T: depth>=4.5mm 且 |Δdepth|<0.2mm 且 xy<=1.5mm 且 |fz|>=1.5N")
    print("      且 |fxy|<=0.45N 且 |t|<=0.05Nm 连续3拍 (已去掉 angle_z)")
    print("F: |f|>50N 或 |t|>5Nm 或出工作空间 或 接近阶段连续回退")
    print("注意: 录制存的是 step 前观测，成功当拍 post-step 未写入。")
    print("      pad_last=True 时复制末帧补这一拍，才是和在线 _state_trans 对齐的评估。")

    # 当前训练集打完整 pad 报告
    main_files = [f for f in files if f.name in ("recorded_data.npz", "recorded_data_filtered.npz")
                  and f.parent.name == "datasets"]
    for f in main_files:
        print("\n########## PAD_LAST 详细报告 ##########")
        print_report(analyze_file(f, pad_last=True))

    print("\n" + "=" * 88)
    print("汇总: 各数据集终态占比  raw vs pad_last")
    print(f"{'file':<32} {'n':>5}  {'raw T':>7} {'raw I':>7} {'raw S':>7}  |  {'pad T':>7} {'pad I':>7} {'pad S':>7} {'pad F':>7}")
    for f in files:
        raw = analyze_file(f, pad_last=False)
        pad = analyze_file(f, pad_last=True)
        if raw["n"] == 0:
            continue
        name = Path(raw["path"]).parent.name + "/" + Path(raw["path"]).name
        if Path(raw["path"]).parent.name == "datasets":
            name = Path(raw["path"]).name
        n = raw["n"]
        cr, cp = raw["counts"], pad["counts"]
        print(f"{name:<32} {n:5d}  "
              f"{100*cr[3]/n:6.1f}% {100*cr[2]/n:6.1f}% {100*cr[1]/n:6.1f}%  |  "
              f"{100*cp[3]/n:6.1f}% {100*cp[2]/n:6.1f}% {100*cp[1]/n:6.1f}% {100*cp[4]/n:6.1f}%")


if __name__ == "__main__":
    main()
