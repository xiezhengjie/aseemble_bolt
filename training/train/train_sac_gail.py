import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import random
import time
import numpy as np
import torch
import gymnasium as gym
from utils import rl_utils, legacy_gail
from utils.checkpoint import has_weight
from algorithm.sac import SACContinuous
from algorithm.discriminator import Discriminator

ENV_ID = "MountainCarContinuous-v0"
N_ENVS = 4
RAW_OBS_DIM = 2


class MountainCarSuccessWrapper(gym.Wrapper):
    """MountainCar 登顶时 terminated=True，写入 info['success'] 供评估。"""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["success"] = bool(terminated)
        return obs, reward, terminated, truncated, info


def make_env():
    """子进程/向量环境工厂：必须是可调用对象，不能传入已创建的 env。"""
    env = gym.make(ENV_ID, render_mode=None, max_episode_steps=1000)
    env = MountainCarSuccessWrapper(env)
    return rl_utils.EpisodeStatsWrapper(env)


def _rewrite_expert_with_dones(path):
    """旧 generate_data 只存 num_episods、不存 dones，补上回合边界。"""
    data = np.load(path, allow_pickle=True)
    if "dones" in data.files:
        return
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    n = int(states.shape[0])
    dones = np.zeros(n, dtype=np.float32)
    if "num_episods" in data.files and n > 0:
        idx = 0
        for L in data["num_episods"]:
            idx += int(L)
            if 0 < idx <= n:
                dones[idx - 1] = 1.0
        if idx != n:
            dones[-1] = 1.0
    elif n > 0:
        dones[-1] = 1.0
    np.savez(path, states=states, actions=actions, dones=dones)
    print(f"[ExpertData] rewrote dones into {path}, n_traj={int(dones.sum())}")


def generate_expert_data(path, n_traj=100, seed=0):
    """没有 npz 时用 bang-bang 专家生成演示（沿速度方向推，可稳定登顶）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID, render_mode=None, max_episode_steps=1000)
    states, actions, dones = [], [], []
    success = 0
    for i in range(n_traj):
        obs, _ = env.reset(seed=int(seed) + i)
        done = False
        reached = False
        while not done:
            action = np.array([1.0 if float(obs[1]) >= 0.0 else -1.0], dtype=np.float32)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            states.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            dones.append(np.float32(done))
            reached = reached or bool(terminated)
            obs = next_obs
        success += int(reached)
    env.close()
    np.savez(
        path,
        states=np.stack(states),
        actions=np.stack(actions),
        dones=np.asarray(dones, dtype=np.float32),
    )
    print(f"[ExpertData] generated {n_traj} trajs, {len(states)} steps, "
          f"success={success}/{n_traj} -> {path}")


def parse_args():
    """消融开关。默认对齐报告 v6：无 BC、无 Trick1、GASILfD + z-score、纯 GAIL。"""
    p = argparse.ArgumentParser(description="SAC+GAIL MountainCar，GASILfD 消融")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bc", action="store_true", help="加载 BC 预训练策略")
    p.add_argument("--prefill", action="store_false", help="Trick 1: buffer_g 预填充专家样本")
    p.add_argument("--gasilf", action=argparse.BooleanOptionalAction, default=True,
                   help="Trick 2: GASILfD，env_return>阈值的轨迹写入专家池")
    p.add_argument("--znorm", action=argparse.BooleanOptionalAction, default=True,
                   help="Trick 3: GAIL 奖励 z-score 归一化")
    p.add_argument("--env-reward-weight", type=float, default=0.0,
                   help="环境奖励混合系数，0=纯 GAIL，0.3=v4 混合")
    p.add_argument("--gasilf-threshold", type=float, default=90.0,
                   help="GASILfD setpoint：env_return 大于该值才入专家池")
    p.add_argument("--gasilf-cap", type=int, default=200,
                   help="GASILfD 最多追加的自模仿轨迹条数")
    p.add_argument("--prefill-count", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    root_dir = rl_utils.find_project_root()
    save_data_dir = root_dir / "datasets"
    save_data_file_name = "mountaincar_expert_data.npz"
    expert_model_dir = root_dir / "models" / "bc_model"
    save_model_dir = root_dir / "models" / "sac_gail_model"
    log_dir = root_dir / "logs" / "sac_gail_log"
    save_data_dir.mkdir(parents=True, exist_ok=True)
    save_model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    expert_path = save_data_dir / save_data_file_name
    if not expert_path.is_file():
        generate_expert_data(expert_path, n_traj=100, seed=seed)
    else:
        _rewrite_expert_with_dones(expert_path)

    env = gym.vector.SyncVectorEnv([make_env] * N_ENVS)
    eval_env = make_env()

    obs_shape = env.single_observation_space.shape
    state_dim = int(np.prod(obs_shape))
    action_dim = int(env.single_action_space.shape[0])
    action_space = env.single_action_space
    env.single_action_space.seed(seed)
    env.single_observation_space.seed(seed)

    # SAC 超参
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    gamma = 0.99
    tau = 0.005
    alpha = 1.0
    alpha_min = 0.08

    # 判别器超参
    disc_lr = 1e-4
    weight_decay = 1e-4
    smoothing = 0.1
    grad_clip_norm = None
    reward_clip = 5.0
    reward_scale = 0.5

    policy_hidden_dim = [256, 256]
    disc_hidden_dim = [64, 64]
    use_orthogonal_init = False

    batch_size = 256
    disc_epochs = 2
    total_timesteps = 50000
    buffer_size = 200000
    learning_starts = 1000
    train_freq = 32
    gradient_steps = 32
    target_network_frequency = 2
    policy_network_frequency = 1
    disc_update_freq = 2

    eval_interval = 5000
    eval_episodes = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_bc = bool(args.bc)
    use_trick1_prefill = bool(args.prefill)
    use_gasilf = bool(args.gasilf)
    gail_reward_runnorm = bool(args.znorm)
    env_reward_weight = float(args.env_reward_weight)
    gasilf_success_threshold = float(args.gasilf_threshold)
    gasilf_add_cap = int(args.gasilf_cap)
    pre_fill_count = int(args.prefill_count)

    ablation_tag = (
        f"{'bc' if use_bc else 'nobc'}"
        f"_{'t1' if use_trick1_prefill else 'not1'}"
        f"_{'gasilf' if use_gasilf else 'nogasilf'}"
        f"_{'znorm' if gail_reward_runnorm else 'noz'}"
        f"_w{env_reward_weight:g}"
    )
    run_name = f"sac_gail_{ablation_tag}__{seed}__{int(time.time())}"
    print(
        f"[SAC-GAIL] device={device} n_envs={N_ENVS} "
        f"state_dim={state_dim} action_dim={action_dim}"
    )
    print(
        f"[ablation] BC={use_bc} Trick1={use_trick1_prefill} "
        f"GASILfD={use_gasilf}(thr={gasilf_success_threshold:g},cap={gasilf_add_cap}) "
        f"z-norm={gail_reward_runnorm} w={env_reward_weight:g}"
    )
    print(f"[run] {run_name}")

    buffer_r = legacy_gail.ReplayBuffer(buffer_size)
    buffer_g = legacy_gail.TrajectoryBuffer(buffer_size)
    buffer_e_manager = rl_utils.ExpertDataManager(capacity=buffer_size)  # 兼容签名：capacity 关键字
    buffer_e_manager.load_data(expert_path, frame_stack=1, raw_obs_dim=RAW_OBS_DIM)
    buffer_e = buffer_e_manager.buffer
    if buffer_e.size() == 0:
        raise FileNotFoundError(f"专家数据为空: {expert_path}")

    expert_state, _ = buffer_e.data()
    running_ms = rl_utils.RunningMeanStd(obs_shape)
    running_ms.update(expert_state)

    if use_trick1_prefill:
        n_fill = 0
        for s, a in buffer_e.iter_trajectories():
            buffer_g.add_trajectory(s, a)
            n_fill += int(s.shape[0])
            if n_fill >= pre_fill_count:
                break
        print(f"[Trick 1] buffer_g pre-filled with {buffer_g.size()} expert steps.")
    else:
        print("[Trick 1] disabled, buffer_g starts empty.")

    agent = SACContinuous(
        state_dim, policy_hidden_dim, action_dim, action_space,
        actor_lr, critic_lr, alpha_lr, tau, gamma, alpha,
        target_network_frequency, policy_network_frequency,
        max_steps=total_timesteps, log_std_init=-3, device=device,
        autotune=True, use_sde=True, use_orthogonal_init=use_orthogonal_init,
        alpha_min=alpha_min,
    )

    disc = Discriminator(
        state_dim, action_dim, disc_hidden_dim, disc_lr, smoothing, weight_decay,
        disc_epochs, batch_size, max_steps=total_timesteps, device=device,
        use_orthogonal_init=use_orthogonal_init,
        grad_clip_norm=grad_clip_norm,
        reward_clip=reward_clip, reward_scale=reward_scale,
    )

    if use_bc:
        if has_weight(expert_model_dir, "policy_net"):
            agent.load_policy(expert_model_dir)
            print(f"[SAC-GAIL] loaded BC policy from {expert_model_dir}")
        else:
            raise FileNotFoundError(
                f"--bc 已开但找不到 BC 权重: {expert_model_dir}")
    else:
        print("[SAC-GAIL] no BC policy, train from random init")

    legacy_gail.train_sac_gail(
        env, eval_env, agent, disc, total_timesteps, train_freq, gradient_steps,
        buffer_r, buffer_g, buffer_e, learning_starts, batch_size, seed=seed,
        eval_interval=eval_interval, eval_episodes=eval_episodes,
        is_save_model=True, running_ms=running_ms,
        save_model_dir=save_model_dir, log_dir=log_dir,
        env_reward_weight=env_reward_weight,
        disc_update_freq=disc_update_freq,
        gail_reward_runnorm=gail_reward_runnorm,
        run_name=run_name,
        use_gasilf=use_gasilf,
        gasilf_success_threshold=gasilf_success_threshold,
        gasilf_add_cap=gasilf_add_cap,
    )


if __name__ == "__main__":
    main()
