"""对指定 checkpoint 做 n=50 固定种子评估（确定性环境）。"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/tmp/work/training")
from pathlib import Path
from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from utils import rl_utils
from algorithm.residual_sac import ResidualSAC

ckpt = sys.argv[1]; tag = sys.argv[2]
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
agent.load_model(ckpt)

N_EP, SEED_OFFSET = 50, 100000
recs = []
for ep in range(N_EP):
    obs, _ = env.reset(seed=SEED_OFFSET + ep)
    done = False; info = {}
    while not done:
        a = agent.take_action(np.asarray(obs, dtype=np.float32), deterministic=True)
        obs, r, te, tr, info = env.step(a)
        done = bool(te or tr)
    succ = bool(info.get("success", False))
    recs.append(succ)
    if (ep+1) % 10 == 0:
        print(f"  [{ep+1}/{N_EP}] succ={sum(recs)}", flush=True)
n = sum(recs)
out = dict(ckpt=ckpt, n_episodes=N_EP, seed_offset=SEED_OFFSET, success_rate=n/N_EP, records=recs)
with open(f"/mnt/agents/output/eval_50ep/{tag}.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"[{tag}] success={n}/{N_EP} = {100*n/N_EP:.0f}%")
