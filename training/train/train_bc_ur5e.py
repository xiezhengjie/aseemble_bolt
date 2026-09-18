import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from algorithm.sac import BehaviorClone, SACContinuous
from utils.rl_utils import (
    ExpertDataManager, find_project_root, wrap_frame_stack,
)


def train(agent, state_dim, train_loader, val_loader, epochs, log_dir, model_dir):
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    run_name = f"bc_ur5e__{int(time.time())}"

    writer = SummaryWriter(Path(log_dir) / run_name)
    train_states, train_actions = train_loader.states, train_loader.actions
    val_states, val_actions = val_loader.states, val_loader.actions
    # 专家数据已是物理归一化（recorded_data_norm.npz），与环境观测同尺度

    batch_size = train_loader.batch_size
    best_state = None
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=epochs)
    except ImportError:
        pbar = None

    for i in range(epochs):
        train_loss = agent.fit_epoch(train_states, train_actions, batch_size)
        agent.lr_decay(i)
        val_loss = agent.eval_epoch(val_states, val_actions)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        writer.add_scalar("train_loss", train_loss, i)
        writer.add_scalar("val_loss", val_loss, i)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in agent.policy.state_dict().items()}
        # 避免每个 epoch 写盘；每 20 轮或结束时落盘当前 best
        if best_state is not None and ((i + 1) % 20 == 0 or i + 1 == epochs):
            torch.save(best_state, model_dir / "policy_net.pt")

        if pbar is not None:
            pbar.set_postfix(Epochs='%d' % (i+1),
                             train_loss='%.4f' % train_loss,
                             val_loss='%.4f' % val_loss,
                             best_val='%.4f' % best_val_loss,
                             improved='*' if improved else '')
            pbar.update(1)

    writer.close()
    return train_loss_list, val_loss_list


if __name__ == "__main__":
    from gymnasium.spaces import Box

    root_dir = find_project_root()
    save_data_dir = root_dir / "datasets"
    save_data_file_name = "recorded_data_norm.npz"
    save_model_dir = root_dir / "models" / "bc_model_ur5e"
    log_dir = root_dir / "logs" / "bc_ur5e_log"
    xml_path = str(root_dir/"mjcf/ur5e_assemble_sence.xml")
    urdf_path = str(root_dir/"urdf/ur5e_assemble.urdf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"[BC] device={device} {gpu_name}".rstrip())

    # 单帧 10 维经 FrameStackObservation 叠成 (T, 10)，GRU 吃时序
    frame_stack = 8
    gru_hidden_dim = 64
    use_h_norm = True  # 基座输入 [s; h_norm]：LayerNorm 随 BC 联训、RL 阶段冻结（铁律 2）
    obs_shape = (frame_stack, 10)
    state_dim = int(np.prod(obs_shape))
    action_dim = 6
    action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    hidden_dim = [512, 512]
    lr = 1e-3
    split = 0.8
    # MX330 kernel launch 开销大，BC 用大 batch 才能跑满；47k 样本 2048 约 18 个 train batch
    batch_size = 2048
    epochs = 1000
    weight_decay = 6e-2
    use_orthogonal_init = False
    log_std_init = -3.67
    use_sde = True
    clip_mean = 2.0

    expert_path = save_data_dir / save_data_file_name
    if not expert_path.is_file():
        raise FileNotFoundError(
            f"缺少归一化专家数据: {expert_path}\n"
            "请先运行: python scripts/normalize_expert_data.py")

    buffer_manager = ExpertDataManager()
    buffer_manager.load_data(expert_path, frame_stack=frame_stack)
    train_loader, val_loader = buffer_manager.trans_dataloder(
        split=split, batch_size=batch_size, device=device)
    print(f"[BC] samples={buffer_manager.buffer.size()} "
          f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
          f"data={expert_path.name}")

    agent_kw = dict(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        action_space=action_space, epochs=epochs, lr=lr, weight_decay=weight_decay,
        device=device, log_std_init=log_std_init, use_sde=use_sde,
        use_orthogonal_init=use_orthogonal_init, clip_mean=clip_mean,
        seq_len=frame_stack, raw_obs_dim=10, gru_hidden_dim=gru_hidden_dim,
        use_h_norm=use_h_norm,
    )
    agent = BehaviorClone(**agent_kw)

    save_model_dir.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    train_loss_list, val_loss_list = train(
        agent, state_dim, train_loader, val_loader, epochs, log_dir, save_model_dir)

    # 测试专家模型（训练时不加载 MuJoCo）；环境已输出归一化观测
    from envs.assemble_mujoco_env import AssembleMuJoCoEnv
    _env = AssembleMuJoCoEnv(xml_path=xml_path, urdf_path=urdf_path, render_mode=None,
                             max_episodic_steps=600)
    _env = wrap_frame_stack(_env, frame_stack)
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    gamma = 0.99
    tau = 0.005
    alpha = 1.0
    alpha_min = 0.08
    target_entropy = - action_dim
    sac_kw = dict(
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr, tau=tau, gamma=gamma,
        alpha=alpha, target_network_frequency=1, policy_frequency=1, max_steps=600,
        log_std_init=-3.67, device=device, autotune=True, use_sde=True,
        use_orthogonal_init=use_orthogonal_init, alpha_min=alpha_min,
        target_entropy=target_entropy,
        seq_len=frame_stack, raw_obs_dim=10, gru_hidden_dim=gru_hidden_dim,
        use_h_norm=use_h_norm,
    )
    sac_agent = SACContinuous(state_dim, hidden_dim, action_dim, action_space, **sac_kw)

    sac_agent.load_policy(save_model_dir)
    random_delta = None
    n_episode = 10
    success_count = 0
    returns = []
    for ep in range(n_episode):
        obs, _ = _env.reset(options={'random_delta': random_delta})
        total_return = 0.0
        done = False
        while not done:
            action = sac_agent.take_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = _env.step(action)
            total_return += reward
            done = terminated or truncated
            if info["success"]:  # 成功
                success_count += 1
        returns.append(total_return)
    print(f"success_rate={success_count/n_episode:.1%}, avg_return={np.mean(returns):.2f}")
