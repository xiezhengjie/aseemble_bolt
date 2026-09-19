"""IK 确定性修复验证：同 seed、同 BC 策略，连跑两遍 20 回合 rollout，
要求逐帧观测、逐回合成功标志完全一致（修复前：20% vs 45%）。"""
import os, sys, time, hashlib
import numpy as np
import torch

sys.path.insert(0, "/tmp/work/training")
from pathlib import Path
from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from utils import rl_utils
from algorithm.residual_sac import ResidualSAC

root_dir = Path("/tmp/work")
xml_path = str(root_dir / "mjcf/ur5e_assemble_sence.xml")
urdf_path = str(root_dir / "urdf/ur5e_assemble.urdf")
bc_dir = root_dir / "models" / "bc_model_ur5e"
FRAME_STACK = 8
N_EP = 20
SEED_OFFSET = 100000

def make_env():
    env = AssembleMuJoCoEnv(xml_path=xml_path, urdf_path=urdf_path,
                            render_mode=None, max_episodic_steps=400)
    env = rl_utils.wrap_frame_stack(env, FRAME_STACK)
    return rl_utils.EpisodeStatsWrapper(env)

torch.manual_seed(0); np.random.seed(0)
env = make_env()
print("[backend] UR5E_IK_BACKEND =", os.environ.get("UR5E_IK_BACKEND", "(default pin)"))

action_space = env.action_space
agent = ResidualSAC(
    base_model_dir=bc_dir, raw_obs_dim=10, action_dim=int(action_space.shape[0]),
    seq_len=FRAME_STACK, gru_hidden_dim=64,
    action_low=action_space.low, action_high=action_space.high,
    hidden_dim=(512, 512), residual_scale=0.05,
    device=torch.device("cpu"),
)

def run_pass():
    per_ep = []
    t0 = time.time()
    for ep in range(N_EP):
        obs, _ = env.reset(seed=SEED_OFFSET + ep)
        done = False
        h = hashlib.sha256()
        ep_ret, ep_len, succ = 0.0, 0, False
        while not done:
            a, _h = agent.take_action_base_only(np.asarray(obs, dtype=np.float32)[None])
            obs, r, term, trunc, info = env.step(a[0])
            h.update(np.ascontiguousarray(obs, dtype=np.float64).tobytes())
            ep_ret += float(r); ep_len += 1
            done = bool(term or trunc)
            succ = bool(info.get("success", False) or (isinstance(info, dict) and info.get("is_success", False)))
        per_ep.append((succ, ep_len, round(ep_ret, 6), h.hexdigest()[:16]))
    dt = time.time() - t0
    return per_ep, dt

# EpisodeStatsWrapper 的 success 字段名确认
probe, _ = env.reset(seed=SEED_OFFSET)
a, _ = agent.take_action_base_only(np.asarray(probe, dtype=np.float32)[None])
o, r, te, tr, info = env.step(a[0])
print("[info keys]", sorted(info.keys()) if isinstance(info, dict) else type(info))

p1, dt1 = run_pass()
p2, dt2 = run_pass()

s1 = [x[0] for x in p1]; s2 = [x[0] for x in p2]
print(f"\n[run1] success={sum(s1)}/{N_EP} ({100*sum(s1)/N_EP:.0f}%)  time={dt1:.1f}s")
print(" ", [(i, x[0], x[1], x[3]) for i, x in enumerate(p1)])
print(f"[run2] success={sum(s2)}/{N_EP} ({100*sum(s2)/N_EP:.0f}%)  time={dt2:.1f}s")
print(" ", [(i, x[0], x[1], x[3]) for i, x in enumerate(p2)])

identical = all(a == b for a, b in zip(p1, p2))
print("\n[verdict] 两遍完全一致:" , identical)
if not identical:
    for i, (a, b) in enumerate(zip(p1, p2)):
        if a != b:
            print(f"  ep{i}: {a} vs {b}")
