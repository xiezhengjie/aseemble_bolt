"""Critic 健康探针：
1) 沿专家演示轨迹，Q(s, a_demo) 是否向成功终点递增（价值传播检查）；
2) 在演示状态上，Q(s, a_base+ã) − Q(s, a_base) 的符号分布（残差方向是否正确）；
3) alpha 当前值（探索是否被 α_min 锁死）。
用法: python3 q_probe.py <ckpt_dir>
"""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/work/training")
from pathlib import Path
from training.model.gail.residual_sac import ResidualSACAgent
from training.common.policy_utils import current_frame

ckpt = sys.argv[1]
root_dir = Path("/tmp/work")
agent = ResidualSACAgent(base_model_dir=root_dir/"models"/"bc_model_ur5e",
                    raw_obs_dim=10, action_dim=6, seq_len=8, gru_hidden_dim=64,
                    action_low=-np.ones(6), action_high=np.ones(6),
                    hidden_dim=(512,512), residual_scale=0.05, device=torch.device("cpu"))
agent.load_model(ckpt)
print(f"[alpha] log_alpha={float(agent.log_alpha):.4f} -> alpha={float(agent.log_alpha.exp()):.4f}")

exp = np.load(root_dir/"datasets"/"recorded_data_norm.npz")
S, A, D = exp["states"].astype(np.float32), exp["actions"].astype(np.float32), exp["dones"].astype(bool)
T = 8
pad = np.zeros((T-1, 10), dtype=np.float32)

def stack(ep):
    ext = np.concatenate([pad, ep], axis=0)
    return np.stack([ext[t:t+T] for t in range(len(ep))])

# 取 3 条演示回合
ends = np.nonzero(D)[0]
prev = 0; episodes = []
for e in ends:
    if e+1-prev >= 30: episodes.append((prev, e+1))
    prev = e+1
    if len(episodes) == 3: break

import torch as th
for i, (a0, a1) in enumerate(episodes):
    st = stack(S[a0:a1])                       # (L,8,10)
    h = agent.base_hidden_batch(st)
    stt = th.as_tensor(st); ht = th.as_tensor(h)
    aa = th.as_tensor(np.clip(A[a0:a1], -1, 1))
    with th.no_grad():
        # h 是冻结 GRU 的归一化表示；critic 只评估残差动作。
        h_n = agent.base_hidden(stt)
        a_base = agent.base_action(stt, h_n)
        s_cur = current_frame(stt, agent.seq_len, agent.raw_obs_dim)
        a_demo_res = aa - a_base
        q1 = agent.critic_1(s_cur, a_demo_res, cond=h_n).reshape(-1).numpy()
        q2 = agent.critic_2(s_cur, a_demo_res, cond=h_n).reshape(-1).numpy()
        q = np.minimum(q1, q2)
        # 残差方向检查：Q(ã_policy) vs Q(0)
        eval_actor = agent.actor_ema if agent.actor_ema is not None else agent.actor
        a_res_pol, _ = eval_actor(s_cur, deterministic=True, cond=h_n)
        a_res_zero = torch.zeros_like(a_res_pol)
        qp1 = agent.critic_1(s_cur, a_res_pol, cond=h_n).reshape(-1).numpy()
        qz1 = agent.critic_1(s_cur, a_res_zero, cond=h_n).reshape(-1).numpy()
    L = len(q)
    seg = np.array_split(np.arange(L), 5)
    qseg = [float(q[idx].mean()) for idx in seg]
    dpol = float((qp1 - qz1).mean())
    print(f"[ep{i}] len={L} Q(demo) 分段均值: {[round(x,3) for x in qseg]} "
          f"(首→尾 Δ={qseg[-1]-qseg[0]:+.3f})  Q(ã_policy)−Q(0)={dpol:+.4f}")
