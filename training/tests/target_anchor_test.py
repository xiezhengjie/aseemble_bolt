"""目标锚定方案对比测试：纯积分 / 对称皮带 / 当前实际位姿锚定。

复现用户报告的三个故障机理并给出对比数据：
  A. 纯旋转 100 步满档      → 目标位置累计位移 / 单步跳变 / 目标领先 / 顶速跟踪
  B. 平移 30 步满档          → 最大目标领先（顶速是否被门限压低）
  C. 正转后反转              → 残留目标位置位移
  D. 接触阻塞持续下压        → 目标-实际最大间隙（积分饱和界）
  E. 接触后满档旋转          → 目标横向拖拽 / 实际横向偏摆峰值（"超调一下又恢复"）

用法: python training/tests/target_anchor_test.py
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from utils.math_utils import rotmat_to_quat, quat_multiply, quat_error_angle, quat_conj

ROOT = Path(__file__).resolve().parents[2]
XML = str(ROOT / "mjcf" / "ur5e_assemble_sence.xml")
URDF = str(ROOT / "urdf" / "ur5e_assemble.urdf")

ORI_SCALE = 0.01          # 与用户实测配置一致
LEASH = 0.003             # 皮带长度 / 条件积分门限 3mm
GAP_ORI = 0.05            # 姿态条件积分门限


# ------------------------------------------------------------------
# 三种 _action_to_pose 变体
# ------------------------------------------------------------------
def variant_pure_integrate(env, action):
    """上一拍目标 + 增量，无任何限幅（会积分饱和）。"""
    if np.allclose(action, np.zeros(6), atol=1e-5):
        return env.last_pos.copy(), env.last_quat.copy()
    dest_pos = env.last_pos + action[:3] * env.pos_action_scale
    w = action[3:] * env.ori_action_scale
    from utils.math_utils import rotvec_to_quat
    dest_quat = quat_multiply(rotvec_to_quat(w), env.last_quat)
    env.last_pos, env.last_quat = dest_pos.copy(), dest_quat.copy()
    return dest_pos, dest_quat


def variant_symmetric_leash(env, action):
    """上一拍目标 + 增量，对称欧氏皮带限幅（目标可被实际位置拖回）。"""
    dest_pos, dest_quat = variant_pure_integrate(env, action)
    cur = env.data.site_xpos[env.eef_site_id]
    gap = dest_pos - cur
    n = np.linalg.norm(gap)
    if n > LEASH and n > 1e-12:
        dest_pos = cur + gap * (LEASH / n)
        env.last_pos = dest_pos.copy()
    return dest_pos, dest_quat


def run_variant(name, patch_fn=None, seed=1234):
    env = AssembleMuJoCoEnv(xml_path=XML, urdf_path=URDF, render_mode=None,
                            ori_action_scale=ORI_SCALE)
    if patch_fn is not None:
        env._action_to_pose = patch_fn.__get__(env)
    env.reset(seed=seed)
    results = {}

    def rollout(actions, label):
        """执行动作序列，逐步记录目标/实际位姿，返回记录字典。"""
        log = {'t_pos': [], 'a_pos': [], 't_quat': [], 'a_quat': [], 'done': False,
               'step_jump': 0.0}
        prev_tpos = env.last_pos.copy()
        for a in actions:
            o, r, term, trunc, info = env.step(a)
            t_pos = env.last_pos.copy()
            a_pos = env.data.site_xpos[env.eef_site_id].copy()
            log['t_pos'].append(t_pos)
            log['a_pos'].append(a_pos)
            log['t_quat'].append(env.last_quat.copy())
            log['a_quat'].append(rotmat_to_quat(
                env.data.site_xmat[env.eef_site_id].reshape(3, 3)))
            log['step_jump'] = max(log['step_jump'],
                                   float(np.linalg.norm(t_pos - prev_tpos)))
            prev_tpos = t_pos
            if term or trunc:
                log['done'] = True
                break
        for k in ('t_pos', 'a_pos'):
            log[k] = np.array(log[k])
        log['label'] = label
        return log

    def rel_z(q_end, q_start):
        # 绕世界 z 的实际转角：相对四元数旋转矢量的 z 分量
        # （初始姿态为 [-180,0,0]，直接取绝对四元数 rotvec 的 z 分量恒为 ~0）
        from utils.math_utils import quat_to_rotvec
        return quat_to_rotvec(quat_multiply(q_end, quat_conj(q_start)))[2]

    def summarize(log):
        t, a = log['t_pos'], log['a_pos']
        lead = np.linalg.norm(t - a, axis=1)
        return {
            '目标累计位移mm': float(np.linalg.norm(t[-1] - t[0])) * 1e3,
            '单步目标跳变mm': log['step_jump'] * 1e3,
            '最大目标领先mm': float(lead.max()) * 1e3,
            '提前终止': log['done'],
        }

    # ---- A. 纯旋转 100 步满档 (rz) ----
    env.reset(seed=seed)
    act = [np.array([0, 0, 0, 0, 0, 1.0])] * 100
    logA = rollout(act, 'A纯旋转')
    r = summarize(logA)
    r['实际旋转量rad'] = rel_z(logA['a_quat'][-1], logA['a_quat'][0])
    r['指令旋转量rad'] = 100 * ORI_SCALE
    # 满档起始 20 步内实际横向偏摆峰值（"超调一下又恢复"指标）
    dev = np.linalg.norm(logA['a_pos'][:20] - logA['a_pos'][0], axis=1)
    r['起始20步实际偏摆峰值mm'] = float(dev.max()) * 1e3
    results['A_纯旋转100步'] = r

    # ---- B. 平移 30 步满档 (-z) + 20 步静止收敛 ----
    env.reset(seed=seed)
    logB = rollout([np.array([0, 0, -1.0, 0, 0, 0])] * 30 + [np.zeros(6)] * 20, 'B平移')
    r = summarize(logB)
    r['静止后残余误差mm'] = float(np.linalg.norm(
        logB['t_pos'][-1] - logB['a_pos'][-1])) * 1e3
    results['B_平移30步'] = r

    # ---- C. 正转 50 步后反转 50 步 ----
    env.reset(seed=seed)
    logC = rollout([np.array([0, 0, 0, 0, 0, 1.0])] * 50
                   + [np.array([0, 0, 0, 0, 0, -1.0])] * 50, 'C正反转')
    r = summarize(logC)
    r['残留姿态误差deg'] = np.degrees(quat_error_angle(logC['t_quat'][-1],
                                                        logC['a_quat'][-1]))
    r['残留实际转角rad'] = rel_z(logC['a_quat'][-1], logC['a_quat'][0])
    results['C_正反转'] = r

    # ---- D. 接触阻塞：下压到底后持续 60 步 ----
    env.reset(seed=seed)
    logD = rollout([np.array([0, 0, -1.0, 0, 0, 0])] * 120, 'D阻塞')
    gap = np.linalg.norm(logD['t_pos'] - logD['a_pos'], axis=1)
    results['D_接触阻塞120步'] = {
        '目标-实际最大间隙mm': float(gap.max()) * 1e3,
        '最终间隙mm': float(gap[-1]) * 1e3,
        '提前终止': logD['done'],
    }

    # ---- E. 下压接触后满档旋转 40 步 ----
    env.reset(seed=seed)
    press = [np.array([0, 0, -1.0, 0, 0, 0])] * 55
    rot = [np.array([0, 0, -0.15, 0, 0, 1.0])] * 40
    logE = rollout(press + rot, 'E接触旋转')
    p0 = logE['a_pos'][len(press) - 1]
    t0 = logE['t_pos'][len(press) - 1]
    lat = np.linalg.norm(logE['a_pos'][len(press):] - p0, axis=1)
    tdrift = np.linalg.norm(logE['t_pos'][len(press):] - t0, axis=1)
    results['E_接触后旋转'] = {
        '旋转期目标横向位移mm': float(tdrift.max()) * 1e3,
        '旋转期实际横向偏摆峰值mm': float(lat.max()) * 1e3,
        '旋转期最大目标领先mm': float(np.linalg.norm(
            logE['t_pos'][len(press):] - logE['a_pos'][len(press):], axis=1).max()) * 1e3,
        '实际旋转量rad': rel_z(logE['a_quat'][-1], logE['a_quat'][len(press) - 1]),
        '旋转期姿态误差饱和界rad': float(np.degrees(max(
            quat_error_angle(t, a) for t, a in zip(logE['t_quat'][len(press):],
                                                   logE['a_quat'][len(press):])))),
    }
    env.close()
    return results


def fmt(v):
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)


if __name__ == '__main__':
    variants = [
        ('纯积分(无抗饱和)', variant_pure_integrate),
        ('对称皮带3mm', variant_symmetric_leash),
        ('当前实际位姿+增量', None),       # None = 使用 env 当前实现
    ]
    all_res = {}
    for name, fn in variants:
        print(f'\n===== {name} =====', flush=True)
        res = run_variant(name, fn)
        all_res[name] = res
        for scen, r in res.items():
            print(f'  {scen}:')
            for k, v in r.items():
                print(f'    {k}: {fmt(v)}')

    # 汇总表
    print('\n===== 汇总对比 =====')
    scen_keys = next(iter(all_res.values())).keys()
    for scen in scen_keys:
        print(f'\n[{scen}]')
        rows = all_res.values()
        metrics = next(iter(rows))[scen].keys()
        for m in metrics:
            vals = [f'{name}: {fmt(r[scen][m])}' for name, r in all_res.items()]
            print(f'  {m:28s} ' + ' | '.join(vals))
