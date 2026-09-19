"""判别性探针：critic 的动作方向导数是否指向专家修正方向？
在距成功终点 k 步的演示状态上，沿 d=(a_demo−a_base) 与随机正交方向扫 Q(s,h,t·d̂)。
若 Q 在 t>0 侧上升 → critic 动作面可识别（瓶颈在预算/探索）；
若平坦/反向 → critic 动作面未识别（瓶颈在动作覆盖）。
用法: python3 q_dir_probe.py <ckpt_dir>
"""
import sys, numpy as np, torch as th
sys.path.insert(0, "/tmp/work/training")
from pathlib import Path
from algorithm.residual_sac import ResidualSAC

ckpt = sys.argv[1]
root_dir = Path("/tmp/work")
agent = ResidualSAC(base_model_dir=root_dir/"models"/"bc_model_ur5e",
                    raw_obs_dim=10, action_dim=6, seq_len=8, gru_hidden_dim=64,
                    action_low=-np.ones(6), action_high=np.ones(6),
                    hidden_dim=(512,512), residual_scale=0.05, device=th.device("cpu"))
agent.load_model(ckpt)
print(f"[alpha]={float(agent.log_alpha.exp()):.5f}")

exp = np.load(root_dir/"datasets"/"recorded_data_norm.npz")
S, A, D = exp["states"].astype(np.float32), exp["actions"].astype(np.float32), exp["dones"].astype(bool)
T = 8; pad = np.zeros((T-1,10), dtype=np.float32)
def stack(ep):
    ext = np.concatenate([pad, ep], axis=0)
    return np.stack([ext[t:t+T] for t in range(len(ep))])

ends = np.nonzero(D)[0]; prev = 0; eps = []
for e in ends:
    if e+1-prev >= 60: eps.append((prev, e+1))
    prev = e+1
    if len(eps) == 8: break

rng = np.random.default_rng(0)
ts = np.array([-0.05, -0.025, 0.0, 0.025, 0.05])
K = 30  # 距终点 30 步
slope_d, slope_r = [], []
for (a0,a1) in eps:
    idx = a1 - 1 - K
    st = stack(S[a0:a1])[[idx - a0]]
    ht = th.as_tensor(agent.base_hidden_batch(st)); stt = th.as_tensor(st)
    with th.no_grad():
        h_n = agent.base_hidden(stt); a_base = agent.base_action(stt, h_n)
        s_cur = agent._s_frame(stt)
    d = (np.clip(A[idx],-1,1) - a_base.numpy().reshape(-1))
    nd = np.linalg.norm(d)
    if nd < 1e-6: continue
    d = d/nd
    r = rng.normal(size=6); r = r - (r@d)*d; r /= np.linalg.norm(r)
    qd, qr = [], []
    with th.no_grad():
        for t in ts:
            qd.append(float(agent.critic_1(s_cur, ht, th.as_tensor((t*d).reshape(1,-1)).float())))
            qr.append(float(agent.critic_1(s_cur, ht, th.as_tensor((t*r).reshape(1,-1)).float())))
    qd = np.array(qd); qr = np.array(qr)
    slope_d.append((qd[-1]-qd[0])/0.1)   # ΔQ/Δt 沿专家方向
    slope_r.append((qr[-1]-qr[0])/0.1)
slope_d = np.array(slope_d); slope_r = np.array(slope_r)
print(f"沿专家修正方向 ∂Q/∂t: mean={slope_d.mean():+.4f}  各回合 {[round(x,3) for x in slope_d]}")
print(f"沿随机正交方向 ∂Q/∂t: mean={slope_r.mean():+.4f}  各回合 {[round(x,3) for x in slope_r]}")
print(f"专家方向 > 随机方向 的回合: {int((slope_d > slope_r).sum())}/{len(slope_d)}")
