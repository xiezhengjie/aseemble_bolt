import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from algorithm.sac import BehaviorClone
from utils.rl_utils import ExpertDataManager, RunningMeanStd, find_project_root

def _mean_loss(losses):
    if not losses:
        return 0.0
    if torch.is_tensor(losses[0]):
        return float(torch.stack(losses).mean())
    return float(np.mean(losses))


def train(agent, state_dim, train_loader, val_loader, epochs, log_dir, model_dir, use_obs_normalized):
    """ 训练 """
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    run_name = f"bc__{int(time.time())}"

    writer = SummaryWriter(log_dir/run_name)
    running_ms = RunningMeanStd(state_dim) if use_obs_normalized else None

    # 用训练集统计量初始化归一化器
    if running_ms is not None:
        for observations, _ in train_loader:
            running_ms.update(observations)

    patience_counter = 0
    with tqdm(total=epochs) as pbar:
        for i in range(epochs):
            train_loss = []
            val_loss = []
            # 训练(不放回式抽样)
            for observations, actions in train_loader:
                if running_ms is not None:
                    observations = running_ms.normalize(observations)
                loss = agent.update(observations, actions)
                train_loss.append(loss)
            train_loss = _mean_loss(train_loss)
            agent.lr_decay(i)

            # 验证（使用训练集上学到的 mean/std 归一化）
            for observations, actions in val_loader:
                if running_ms is not None:
                    observations = running_ms.normalize(observations)
                loss = agent.validation(observations, actions)
                val_loss.append(loss)
            val_loss = _mean_loss(val_loss)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            writer.add_scalar("train_loss", train_loss, i)
            writer.add_scalar("val_loss", val_loss, i)

            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                agent.save_model(model_dir)
                if running_ms is not None:
                    running_ms.save_normalizer(model_dir)
                patience_counter = 0             # 重置耐心计数
            else:
                patience_counter += 1            # 验证 Loss 没改善
            if patience_counter >= 100:
                print(f"Early stopping at epoch {i+1}")
                break

            pbar.set_postfix({'Epochs': '%d' % (i+1), 'train_loss': '%.4f' % train_loss, 'val_loss': '%.4f' % val_loss})
            pbar.update(1)

    writer.close()
    return train_loss_list, val_loss_list


if __name__=="__main__":
    root_dir = find_project_root()
    save_data_dir = root_dir/"datasets"
    save_data_file_name = "mountaincar_expert_data.npz"
    save_model_dir = root_dir/"models"/"bc_model" 
    log_dir = root_dir/"logs/bc_log"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make("MountainCarContinuous-v0")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    action_space = env.action_space

    hidden_dim = [256, 256]

    lr = 1e-3
    split = 0.8
    batch_size = 128
    epochs = 500
    weight_decay = 6e-2

    use_obs_normalized = True   
    use_orthogonal_init = False

    # 从专家数据库中读取专家数据
    buffer_manager = ExpertDataManager()
    buffer_manager.load_data(save_data_dir/save_data_file_name)
    train_loader, val_loader = buffer_manager.trans_dataloder(
        split=split, batch_size=batch_size, device=device)

    agent =  BehaviorClone(state_dim=state_dim, 
                           action_dim=action_dim, 
                           hidden_dim=hidden_dim, 
                           action_space=action_space, 
                           epochs=epochs,
                           lr=lr, 
                           weight_decay=weight_decay,
                           device=device, 
                           log_std_init=-3.67, 
                           use_sde=True,
                           use_orthogonal_init=use_orthogonal_init)
    
    # 训练
    train_loss_list, val_loss_list = train(agent, state_dim, train_loader, val_loader, epochs, log_dir, save_model_dir, use_obs_normalized)

    agent.load_model(model_dir=save_model_dir)

    # 测试专家模型（MountainCar 无 success/random_delta 标记，做独立评估）
    running_ms = RunningMeanStd(state_dim)
    running_ms.load_normalizer(model_dir=save_model_dir)
    n_episode = 10
    success_count = 0
    returns = []
    for ep in range(n_episode):
        obs, _ = env.reset()
        total_return = 0.0
        done = False
        while not done:
            action = agent.take_action(running_ms.normalize(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            done = terminated or truncated
            if obs[0] >= 0.45:  # MountainCar 到达山顶即成功
                success_count += 1
        returns.append(total_return)
    print(f"success_rate={success_count/n_episode:.1%}, avg_return={np.mean(returns):.2f}")