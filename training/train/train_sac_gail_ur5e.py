"""修正版残差 SAC-GAIL（结局条件 (s,s′) 判别器 + SQIL 化/纯稀疏奖励）。

相对 train_sac_gail_ur5e.py 的修正链（每一环均有本地实测依据）：
  1. 判别器输入 (s,a) → (s,s′)：消除"孔口奖励谷地"（实测 (s,a) 型 r̃ 与
     xy 误差 spearman=+0.70，稠密奖励与任务进展反向）；
  2. α 同步缩放：奖励 ÷20 后 α 必须同步 ÷20，否则熵项接管 actor；
  3. γ 0.99 → 0.997：有效视野 100→333 步，R_succ 能传回接近/搜孔阶段；
  4. 成功回合尾部 32 步上采样（每 batch 12.5%）：稀疏成功信号的接力棒；
  5. v4 奖励尺度修正：r̃ 不再 ÷20（标准 GAIL softplus(−z)，clamp ±4）。
     v3 实测：近专家数据上 D 的 logit 只有 ~±1，÷20 后有效对比度
     ~0.01-0.05/步，被熵项 α·logπ≈1.2/步 淹没 → 成功率封顶在 BC 基线。
     修正后对比度 ~1-2/步 vs 熵项 ≤0.54/步（α_max=0.02），奖励占主导；
     R_succ 同步取 r̃_max·γ/(1−γ)≈1335，critic 换 Huber(β=10) 防尖峰；
  6. v5 结局条件判别器：正类 = 专家 + 在线成功回合的 (s,s+K) 对，
     负类 = 最近窗口失败回合对。v4 实测：纯专家/生成二分类的置信度由
     "区域可分性"决定（接近带 r̃≈0.4 >> 孔区 r̃≈0.1），奖励放大后策略
     离开孔口反复"接近"务农（xy 误差 2mm→17mm、深度变负、成功率反降）。
     结局条件后 D 学"演化是否通向成功"：务农环整体落入负类 → 自我纠正；
     成功回合的接近段与孔区段同为正类 → 区域反转结构性消除；
  7. v6 奖励 EMA：r̃ 由 D 的 EMA 副本（τ=0.99/次更新）产出。v5 实测
     D 与策略互相追逐形成极限环（D margin 0.52↔0.73 摆动，成功率
     50%↔34% 跟随振荡）；EMA 把奖励地貌时间常数拉到 ~20 轮，阻尼追逐；
  8. v7 SQIL 化：r̃ = C·1{z_EMA<0}，C=softplus(4)≈4.018。v4-v6 的失败
     根源不是边界本身，而是奖励幅度携带"区域置信度"并随 D 漂移；常数化后
     无可务农幅度，奖励漂移被限幅到 ±C；
  9. v8 纯稀疏对照（SS_SPARSE_ONLY=1）：r = 1{success}，关闭 D 更新。
     对齐 ResiP/SERL 的稳定做法，直接优化折扣成功概率；用于验证在彻底
     移除非平稳 IRL 奖励后，残差 SAC 本身能否稳定超过 BC 基线；
 10. v9 评估/部署 actor EMA（SS_ACTOR_EMA，默认 0.995）：在线 actor 继续
     探索，确定性评估与 checkpoint 使用 Polyak 平均权重，阻尼 Q 地貌微调
     造成的策略抖动；v8 实测奖励已平稳但 eval 仍在 40%↔60% 摆动。
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from utils import rl_utils
from utils.checkpoint import has_weight
from algorithm.discriminator import MarkovDiscriminatorSS, build_expert_ss_pairs
from algorithm.residual_sac import ResidualSAC, residual_target_entropy
from envs.assemble_mujoco_env import AssembleMuJoCoEnv

N_ENVS = int(os.environ.get("SS_N_ENVS", "3"))
FRAME_STACK = 8

root_dir = rl_utils.find_project_root()
save_data_dir = root_dir / "datasets"
save_data_file_name = "recorded_data_norm.npz"
expert_model_dir = root_dir / "models" / "bc_model_ur5e"
save_model_dir = root_dir / "models" / "sac_gail_ur5e_ss_model"
log_dir = root_dir / "logs" / "sac_gail_ur5e_ss_log"
xml_path = str(root_dir / "mjcf/ur5e_assemble_sence.xml")
urdf_path = str(root_dir / "urdf/ur5e_assemble.urdf")


def make_env():
    env = AssembleMuJoCoEnv(
        xml_path=xml_path, urdf_path=urdf_path,
        render_mode=None, max_episodic_steps=400,
    )
    env = rl_utils.wrap_frame_stack(env, FRAME_STACK)
    return rl_utils.EpisodeStatsWrapper(env)


def make_worker(idx, base_seed):
    def _thunk(idx=int(idx), base_seed=int(base_seed)):
        env = make_env()
        s = base_seed + idx
        env.action_space.seed(s)
        env.observation_space.seed(s)
        return env
    return _thunk


def parse_args():
    p = argparse.ArgumentParser(description="残差 SAC-GAIL UR5e 插装（(s,s′) 结局条件修正版）")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--residual-scale", type=float, default=0.05)
    p.add_argument("--total-timesteps", type=int, default=400_000)
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--init-model", type=str, default="")
    p.add_argument("--model-dir", type=str, default="",
                   help="模型输出目录；默认 models/sac_gail_ur5e_ss_model")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else save_model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_envs = N_ENVS
    frame_stack = FRAME_STACK
    gru_hidden_dim = 64

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 小内存机器用 SyncVectorEnv（单进程，省每个 worker 的解释器+torch 开销）；
    # 设 SS_ASYNC=1 切回 AsyncVectorEnv（多核大内存机器更快）。
    if os.environ.get("SS_ASYNC", "0") == "1":
        env = gym.vector.AsyncVectorEnv([make_worker(i, seed) for i in range(n_envs)])
    else:
        env = gym.vector.SyncVectorEnv([make_worker(i, seed) for i in range(n_envs)])
    eval_env = make_env()

    obs_shape = env.single_observation_space.shape
    action_dim = int(env.single_action_space.shape[0])
    action_space = env.single_action_space
    env.single_action_space.seed(seed)
    env.single_observation_space.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_only = os.environ.get("SS_SPARSE_ONLY", "0") == "1"

    # ---------------- 超参（修正版） ----------------
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 1e-4
    # γ=0.997 视野 333 步：bootstrap 噪声积累 ∝1/(1−γ)，实测 TD 残差 0.17，
    # 比残差球内动作对比度 0.008 高 20 倍；γ=0.99 把噪声压 3.3 倍
    gamma = float(os.environ.get("SS_GAMMA", "0.997"))
    tau = 0.005
    # 修正5（v4）：奖励不再 ÷20（标准 GAIL softplus(−z)，clamp±4 → r̃≤4.02），
    # α 同步下调到 0.02：熵项 α·logπ≈0.54/步 < 稠密对比度 ~1-2/步，
    # 奖励信号首次在 actor 目标中占主导（v3 实测 0.01 vs 1.2，被封顶）
    alpha = float(os.environ.get("SS_ALPHA", "0.02"))
    alpha_min = float(os.environ.get("SS_ALPHA_MIN", "0.002"))
    alpha_max = float(os.environ.get("SS_ALPHA_MAX", "0.02"))
    # v14：γ=0.997 有效视野 333 步会把每步熵奖励放大 333 倍，
    # α·(−logπ)·333 ≈ 2000α ≫ R_succ=1 时熵塑形淹没成功信号；
    # 且 target_H=−d+d·log ε 会把探索 σ 压到 0.005（探索实际关闭）。
    # 推荐：SS_TARGET_H=-6 SS_ALPHA=0.005 SS_ALPHA_MIN=0.0001
    actor_ema_tau = float(os.environ.get("SS_ACTOR_EMA", "0.995"))

    disc_lr = 1e-3
    # K 步 (s,s′) 对高度可分，R1-GP 会压死 logit margin（实测 GP=1.0 时 BCE 卡 1.30、
    # 输出钉 0.53；GP=0.01 时 acc 0.87、对比度正常）→ 轻正则即可
    disc_gp_coef = float(os.environ.get("SS_GP", "0.01"))
    disc_entropy_coef = 0.0

    policy_hidden_dim = [512, 512]
    disc_hidden_dim = [64, 64]

    batch_size = 512
    sac_updates = int(os.environ.get("SS_SAC_UPDATES", "20"))  # 默认 20/1200 步（UTD 1:60，SERL 常用 1:1~8:1）
    disc_updates = 0 if sparse_only else 5
    disc_batch_size = 512

    residual_scale = float(args.residual_scale)
    _th = os.environ.get("SS_TARGET_H", "")
    target_entropy = (float(_th) if _th else
                      residual_target_entropy(action_dim, residual_scale))
    total_timesteps = int(args.total_timesteps)
    buffer_size = 200_000
    steps_per_iter = n_envs * 400
    buffer_g_window = 3 * steps_per_iter

    # GAIL 模式：R_succ = r̃_max·γ/(1−γ)，保持"早成功 ≥ 任何拖延"的最坏情形界。
    # 纯稀疏模式：r=1{success}，Q 直接回归折扣成功概率，无需稠密奖励尺度配平。
    R_succ = (1.0 if sparse_only else
              math.log1p(math.exp(4.0)) * gamma / (1.0 - gamma))
    success_tail = 32                # 修正4：成功回合尾部上采样
    succ_frac = 0.125
    prefill_steps = steps_per_iter

    eval_interval = 12_000           # 每 10 轮评估
    # n=10 的二项噪声 σ≈13-15%，无法判"稳定上升不掉"；SS_EVAL_EPISODES=20 把 σ 压到 ~9%
    eval_episodes = int(os.environ.get("SS_EVAL_EPISODES", "10"))

    if not has_weight(expert_model_dir, "policy_net"):
        raise FileNotFoundError(f"缺少 BC 基座权重: {expert_model_dir}")

    reward_mode = "sparse" if sparse_only else "gail_binary"
    run_name = (
        f"sac_gail_ur5e_ss__{reward_mode}__eps{residual_scale:g}__"
        f"Rsucc{R_succ:g}__g{gamma:g}__{seed}__{int(time.time())}"
    )
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    print(f"[SAC-GAIL SS] device={device} n_envs={n_envs} mode={reward_mode} γ={gamma} "
          f"R_succ={R_succ:.3g} α∈[{alpha_min},{alpha_max}] target_H={target_entropy:.2f}",
          flush=True)
    print(f"[run] {run_name}", flush=True)

    # ---------------- 数据 ----------------
    buffer_r = rl_utils.ResidualReplayBuffer(
        buffer_size, seq_len=frame_stack, raw_obs_dim=10,
        gru_hidden_dim=gru_hidden_dim, action_dim=action_dim)
    expert_path = save_data_dir / save_data_file_name
    if not expert_path.is_file():
        raise FileNotFoundError(f"缺少归一化专家数据: {expert_path}")
    exp = np.load(expert_path)
    pair_k = int(os.environ.get("SS_PAIR_K", "8"))  # K 步演化窗口（单步不可分→K 步可分）
    expert_ss = build_expert_ss_pairs(exp["states"], exp["dones"], k=pair_k)
    print(f"[data] expert (s,s+{pair_k}) pairs: {len(expert_ss[0])}", flush=True)

    # ---------------- agent / disc ----------------
    agent = ResidualSAC(
        base_model_dir=expert_model_dir,
        raw_obs_dim=10, action_dim=action_dim,
        seq_len=frame_stack, gru_hidden_dim=gru_hidden_dim,
        action_low=action_space.low, action_high=action_space.high,
        hidden_dim=policy_hidden_dim, residual_scale=residual_scale,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau, alpha=alpha, alpha_min=alpha_min,
        alpha_max=alpha_max,
        target_entropy=target_entropy, device=device,
        actor_ema_tau=actor_ema_tau,
    )
    if args.init_model:
        agent.load_model(args.init_model)
        print(f"[init] loaded {args.init_model}", flush=True)

    # ---------------- 演示预填充（SERL/ResiP 式，SS_DEMO_BUFFER=1 开启） ----------------
    # 纯稀疏奖励下 critic 唯一的非零目标来自稀有在线成功；把专家演示作为转移
    # 预充进 SAC 回放池（不改奖励），让价值从成功终点向插入区自举。
    if os.environ.get("SS_DEMO_BUFFER", "0") == "1":
        _s = exp["states"].astype(np.float32)
        _a = np.clip(exp["actions"].astype(np.float32), -1.0, 1.0)
        _d = exp["dones"].astype(bool)
        _T = frame_stack
        _pad = np.zeros((_T - 1, _s.shape[1]), dtype=np.float32)
        _obs_l, _nobs_l, _act_l, _don_l, _suc_l = [], [], [], [], []
        _prev = 0
        for _e in np.nonzero(_d)[0]:
            L = _e + 1 - _prev
            if L < 2:
                _prev = _e + 1
                continue
            ext = np.concatenate([_pad, _s[_prev:_e + 1]], axis=0)   # zero-pad 对齐 wrap_frame_stack
            st = np.stack([ext[t:t + _T] for t in range(L)])
            _obs_l.append(st[:-1]); _nobs_l.append(st[1:])
            _act_l.append(_a[_prev:_e])
            _dd = np.zeros((L - 1, 1), dtype=np.float32); _dd[-1, 0] = 1.0
            _ss = np.zeros((L - 1, 1), dtype=np.float32); _ss[-1, 0] = 1.0
            _don_l.append(_dd); _suc_l.append(_ss)
            _prev = _e + 1
        _obs = np.concatenate(_obs_l); _nobs = np.concatenate(_nobs_l)
        _act = np.concatenate(_act_l); _don = np.concatenate(_don_l); _suc = np.concatenate(_suc_l)
        _h = agent.base_hidden_batch(_obs)
        _hn = agent.base_hidden_batch(_nobs)
        buffer_r.add_batch(_obs, _nobs, _h, _hn, _act,
                           np.zeros((len(_obs), 1), dtype=np.float32), _don, _suc)
        print(f"[demo] preloaded {len(_obs)} expert transitions "
              f"({int(_suc.sum())} success terminals)", flush=True)

    disc = MarkovDiscriminatorSS(
        raw_obs_dim=10, hidden_dim=disc_hidden_dim, lr=disc_lr,
        gp_coef=disc_gp_coef, entropy_coef=disc_entropy_coef,
        device=device,
    )

    rl_utils.train_sac_gail_residual_ss(
        env, eval_env, agent, disc,
        buffer_r, expert_ss,
        total_timesteps, seed,
        sac_updates=sac_updates, disc_updates=disc_updates,
        batch_size=batch_size, disc_batch_size=disc_batch_size,
        success_reward=R_succ,
        disc_reward_coef=0.0 if sparse_only else 1.0,
        buffer_g_window=buffer_g_window,
        pair_k=pair_k,
        success_tail=success_tail, succ_frac=succ_frac,
        eval_interval=eval_interval, eval_episodes=eval_episodes,
        save_model_dir=model_dir, log_dir=log_dir, run_name=run_name,
        is_save_model=True, is_draw=True,
        prefill_steps=prefill_steps, max_steps_lr=total_timesteps,
        steps_per_iter=steps_per_iter,
    )


if __name__ == "__main__":
    main()
