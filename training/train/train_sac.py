import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import json
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
# 强制使用 Agg 后端，避免无显示环境（如沙箱）下 Qt/XCB 崩溃
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import collections
from utils import rl_utils, legacy_gail
from algorithm.sac import SACContinuous
from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from pathlib import Path
import gymnasium as gym

root_dir = rl_utils.find_project_root()
save_data_dir = root_dir/"datasets"

save_data_file_name = "mountaincar_expert_data.npz"


save_model_dir = root_dir/"models"/"sac_model" 
log_dir = root_dir/"logs/sac_log"
draw_path = save_model_dir/"sac_learning_episode_reward_graph.svg"
reward_save_path = save_model_dir/"current_reward.npy"
model_path = str(root_dir/"mjcf/ur5e_assemble_sence.xml")
urdf_path = str(root_dir/"urdf/ur5e_assemble.urdf")
seed = 1
# env = AssembleMuJoCoEnv(xml_path=model_path, urdf_path=urdf_path, render_mode="human")
# eval_env = AssembleMuJoCoEnv(xml_path=model_path, urdf_path=urdf_path, render_mode=None)

env_name = 'MountainCarContinuous-v0'
env = rl_utils.EpisodeStatsWrapper(gym.make(env_name, render_mode=None, max_episode_steps=1000))
eval_env = rl_utils.EpisodeStatsWrapper(gym.make(env_name, render_mode=None, max_episode_steps=1000))

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
env.action_space.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# MountainCarContinuous-v0 用较小网络即可
# hidden_dim = [64, 64]
# batch_size = 512

# actor_lr = 3e-4
# critic_lr = 3e-4
# alpha_lr = 3e-4

# gamma = 0.9999
# tau = 0.01
# alpha = 0.1

# num_episodes = 5000
# buffer_size = 50000
# learning_starts = 0
# train_freq = 32
# gradient_steps = 32
# target_network_frequency=1
# policy_network_frequency=1


hidden_dim = [256, 256]
batch_size = 64

actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4

gamma = 0.99
tau = 0.005
alpha = 1.0  

total_timesteps = 50000
buffer_size = 100000
learning_starts = 1000
train_freq = 32
gradient_steps = 32
target_network_frequency=1  
policy_network_frequency=1

# actor_lr = 3e-4
# critic_lr = 3e-3
# alpha_lr = 3e-4
# num_episodes = 1000
# # hidden_dim = 128
# hidden_dim = [256,256]
# gamma = 0.99
# tau = 0.005  # 软更新参数
# buffer_size = 100000
# minimal_size = 1000
# batch_size = 64

num_trajectorys = 100 # 取100条专家经验数据

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

action_space = env.action_space 

replay_buffer = legacy_gail.ReplayBuffer(int(buffer_size))

# self, state_dim, hidden_dim, action_dim, action_bound,
                #  actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                #  device
agent = SACContinuous(state_dim, hidden_dim, action_dim, action_space,
                      actor_lr, critic_lr, alpha_lr, tau, gamma, alpha,
                      target_network_frequency=target_network_frequency,
                      policy_frequency=policy_network_frequency,
                      device=device, autotune=True,
                      use_sde=True, log_std_init=-3, use_orthogonal_init=False, clip_mean=2.0)

# agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
#                        actor_lr, critic_lr, alpha_lr, target_entropy, tau,
#                        gamma, device)

reward = legacy_gail.train_off_policy_agent(env, eval_env, agent, total_timesteps, train_freq, gradient_steps, replay_buffer,
                                        learning_starts, batch_size, seed, is_save_model=True, normalized_observation=True,
                                        save_model_dir=save_model_dir, log_dir=log_dir)

# # 从保存的 info.json 中恢复观测归一化统计量，用于专家数据生成
# obs_normalizer = None
# info_path = save_model_dir / "final_model" / "info.json"
# if info_path.exists():
#     with open(info_path, 'r', encoding='utf-8') as f:
#         final_info = json.load(f)
#     if 'obs_normalizer' in final_info:
#         obs_normalizer = rl_utils.RunningMeanStd(state_dim)
#         obs_normalizer.load_state_dict(final_info['obs_normalizer'])

# # 保存最终的专家数据（训练时使用归一化状态取动作，保存原始状态）
# exp_data_man = rl_utils.ExpertDataManager(env=env, agent=agent, num_trajectorys=num_trajectorys)
# exp_data_man.generate_data(save_data_dir, save_data_file_name, obs_normalizer=obs_normalizer)

# 绘图
os.makedirs(save_model_dir, exist_ok=True)
plt.plot(reward, color="red")
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("sac_learnning")
plt.savefig(draw_path)
plt.close()


# # 保存当前奖励
# np.save(reward_save_path, reward)
# agent.load_model(save_model_dir/"best_model")

# reward = rl_utils.test_agent(env=env, agent=agent, num_episodes=num_trajectorys)
# plt.plot(reward, color="red")
# plt.xlabel("episode")
# plt.ylabel("reward")
# plt.title("sac_learnning")
# plt.show()
# plt.savefig(draw_path)