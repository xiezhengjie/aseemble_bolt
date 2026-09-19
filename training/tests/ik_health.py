"""统计 BC rollout 中 pinocchio IK 的未收敛率与残差误差分布。"""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/work/training")
from pathlib import Path
import algorithm.ur5e_ik as uik
from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from utils import rl_utils
from algorithm.residual_sac import ResidualSAC

# ---- 插桩：包装 ik_pinocchio 记录失败 ----
stats = {"calls": 0, "fail": 0, "err_fail": []}
orig = uik.UR5eIK.ik_pinocchio
def wrapped(self, tgt_pos, tgt_quat, seed):
    q = orig(self, tgt_pos, tgt_quat, seed)
    stats["calls"] += 1
    # 用 FK 复查到位误差
    pos, quat = self.fk_pinocchio(q)
    from utils.math_utils import quat_to_rotmat
    R_t = quat_to_rotmat(tgt_quat); R_a = quat_to_rotmat(quat)
    pos_err = float(np.linalg.norm(pos - tgt_pos))
    rot_err = float(np.arccos(np.clip((np.trace(R_t.T @ R_a) - 1) / 2, -1, 1)))
    if pos_err > 1e-3 or rot_err > 1e-2:
        stats["fail"] += 1
        stats["err_fail"].append((pos_err, rot_err))
    return q
uik.UR5eIK.ik_pinocchio = wrapped

root_dir = Path("/tmp/work")
env = AssembleMuJoCoEnv(xml_path=str(root_dir/"mjcf/ur5e_assemble_sence.xml"),
                        urdf_path=str(root_dir/"urdf/ur5e_assemble.urdf"),
                        render_mode=None, max_episodic_steps=400)
env = rl_utils.wrap_frame_stack(env, 8)
env = rl_utils.EpisodeStatsWrapper(env)
agent = ResidualSAC(base_model_dir=root_dir/"models"/"bc_model_ur5e",
                    raw_obs_dim=10, action_dim=6, seq_len=8, gru_hidden_dim=64,
                    action_low=env.action_space.low, action_high=env.action_space.high,
                    hidden_dim=(512,512), residual_scale=0.05, device=torch.device("cpu"))

for ep in range(3):
    obs, _ = env.reset(seed=100000+ep)
    done = False
    while not done:
        a, _ = agent.take_action_base_only(np.asarray(obs, dtype=np.float32)[None])
        obs, r, te, tr, info = env.step(a[0])
        done = bool(te or tr)

print(f"IK calls={stats['calls']}  fail(>1mm 或 >0.01rad)={stats['fail']}  rate={stats['fail']/max(1,stats['calls'])*100:.2f}%")
if stats["err_fail"]:
    pe = np.array([e[0] for e in stats["err_fail"]]); re_ = np.array([e[1] for e in stats["err_fail"]])
    print(f"fail pos_err: mean={pe.mean()*1000:.2f}mm max={pe.max()*1000:.2f}mm | rot_err: mean={np.degrees(re_.mean()):.2f}deg max={np.degrees(re_.max()):.2f}deg")
