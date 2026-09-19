"""确定性 IK 下的 BC 基线：50 回合固定种子评估 + 失败分解。"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/tmp/work/training")
from pathlib import Path
from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from utils import rl_utils
from algorithm.residual_sac import ResidualSAC

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

N_EP, SEED_OFFSET = 50, 100000
recs = []
for ep in range(N_EP):
    obs, _ = env.reset(seed=SEED_OFFSET + ep)
    done = False; ep_len = 0
    info = {}
    while not done:
        a, _ = agent.take_action_base_only(np.asarray(obs, dtype=np.float32)[None])
        obs, r, te, tr, info = env.step(a[0])
        ep_len += 1; done = bool(te or tr)
    succ = bool(info.get("success", False))
    if succ: cat = "success"
    elif info.get("fail_contact", False): cat = "fail_contact"
    elif info.get("fail_workspace", False): cat = "fail_workspace"
    else: cat = "timeout"
    recs.append(dict(ep=ep, seed=SEED_OFFSET+ep, success=succ, cat=cat,
                     length=ep_len, state=int(info.get("state", -1)),
                     depth=float(info.get("depth", 0.0))))
    if (ep+1) % 10 == 0:
        print(f"  [{ep+1}/{N_EP}] succ so far: {sum(r['success'] for r in recs)}", flush=True)

n_succ = sum(r["success"] for r in recs)
from collections import Counter
cats = Counter(r["cat"] for r in recs)
out = dict(backend="pin", n_episodes=N_EP, seed_offset=SEED_OFFSET,
           success_rate=n_succ/N_EP, n_success=n_succ,
           fail_breakdown=dict(cats), records=recs)
Path("/mnt/agents/output/eval_50ep").mkdir(parents=True, exist_ok=True)
with open("/mnt/agents/output/eval_50ep/bc_base_pin.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"[BC/pin] success={n_succ}/{N_EP} = {100*n_succ/N_EP:.0f}%  breakdown={dict(cats)}")
