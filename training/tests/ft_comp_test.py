"""惯性补偿验收：无接触大加速度往返运动，校准力应≈0（噪声地板 ~0.01N）。"""
import sys, os, numpy as np
sys.path.insert(0, "/tmp/work/training")
from pathlib import Path
from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from utils.math_utils import rotmat_to_quat
root = Path("/tmp/work")
env = AssembleMuJoCoEnv(xml_path=str(root/"mjcf/ur5e_assemble_sence.xml"),
                        urdf_path=str(root/"urdf/ur5e_assemble.urdf"),
                        render_mode=None, max_episodic_steps=400)
env.reset(seed=123)
ctl = env.ur5e_controller
site = env.eef_site_id
q0 = rotmat_to_quat(env.data.site_xmat[site].reshape(3,3))
p0 = env.data.site_xpos[site].copy()

mode = os.environ.get("UR5E_FT_INERTIA_COMP", "1")
print(f"inertia_comp={mode}, tool_mass={env.model.body_mass[env.eef_body_id]:.3f}kg")

# 大加速度往返：xy 平面 ±20mm 快速扫动（远超 1 m/s^2 的惯性补偿门槛）
stats = []
for i in range(60):
    tgt = p0.copy()
    phase = i % 20
    amp = 0.02 if phase < 10 else -0.02
    tgt[0] += amp; tgt[1] += 0.5*amp
    ctl.update(tgt, q0, admittance_ratio=10, admittance_sub_steps=5)
    ft = ctl.calibrated_ft
    stats.append([np.linalg.norm(ft[:3]), np.linalg.norm(ft[3:])])
    p0 = env.data.site_xpos[site].copy()  # 往返
stats = np.array(stats[10:])  # 去掉起动段
print(f"|F| max={stats[:,0].max():.3f}N mean={stats[:,0].mean():.3f}N | maxFxyz={np.abs(ctl.max_calibrated_ft[:3]).max():.3f}")
print(f"|T| max={stats[:,1].max():.3f}Nm mean={stats[:,1].mean():.3f}Nm")
