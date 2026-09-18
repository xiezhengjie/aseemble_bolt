import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from utils import rl_utils

root_dir = rl_utils.find_project_root()
save_data_dir = root_dir/"datasets"
save_data_file_name = "mountaincar_expert_data.npz"
save_model_dir = root_dir/"models"/"sac_sb3_model"
log_dir = root_dir/"logs/sac_sb3_log"
seed = 1

# 创建训练环境和评估环境
env_name = 'MountainCarContinuous-v0'
venv = DummyVecEnv([lambda: Monitor(gym.make(env_name, render_mode=None, max_episode_steps=1000))])
env = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

eval_venv = DummyVecEnv([lambda: Monitor(gym.make(env_name, render_mode=None, max_episode_steps=1000))])
eval_env = VecNormalize(eval_venv, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=64,
    learning_starts=1000,
    train_freq=32,
    gradient_steps=32,
    use_sde=True,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    # target_update_interval=1,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    seed=seed,
    tensorboard_log=str(log_dir),
)

class SyncedEvalCallback(EvalCallback):
    """评估前把训练环境的归一化统计量同步到评估环境"""
    def _on_step(self) -> bool:
        sync_envs_normalization(self.model.get_vec_normalize_env(), self.eval_env)
        return super()._on_step()

eval_cb = SyncedEvalCallback(
    eval_env,
    best_model_save_path=str(save_model_dir/"best_model"),
    log_path=str(log_dir/"eval"),
    eval_freq=5000,
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

# 训练
total_timesteps = 200000

checkpoint_cb = CheckpointCallback(
    save_freq=total_timesteps//4,
    save_path=str(save_model_dir),
    name_prefix="sac_mountaincar",
    save_vecnormalize=True,  
)
# eval_cb = EvalCallback(
#     eval_env,
#     best_model_save_path=str(save_model_dir/"best_model"),
#     log_path=str(log_dir/"eval"),
#     eval_freq=5000,
#     n_eval_episodes=20,
#     deterministic=True,
#     render=False,
# )
print(f"开始训练 SAC on MountainCarContinuous-v0，总步数 {total_timesteps} ...")
model.learn(total_timesteps=total_timesteps, 
            callback=[checkpoint_cb, eval_cb],
            log_interval=10,
            progress_bar=True,)

# 保存最终模型
model.save(str(save_model_dir/"sac_mountaincar_final"))
print("训练完成，模型已保存")

# 加载最优模型生成专家数据
best_model_path = save_model_dir/"best_model"/"best_model.zip"
if best_model_path.exists():
    model = SAC.load(str(best_model_path), env=env)
    print("已加载 best_model 生成专家数据")
else:
    print("未找到 best_model，使用最终模型生成专家数据")

# 用训练好的专家生成专家轨迹数据
num_trajectorys = 100
states, actions, dones = [], [], []
success_count = 0

for i in range(num_trajectorys):
    obs = env.reset()
    ep_states, ep_actions, ep_dones = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done_vec, info = env.step(action)
        ep_states.append(obs.flatten())
        ep_actions.append(action.flatten())
        ep_dones.append(bool(done_vec[0]))
        obs = next_obs
        done = bool(done_vec[0])
    # 判断是否成功（MountainCar 到达山顶 position >= 0.45）
    if obs[0][0] >= 0.45:
        success_count += 1
    states.extend(ep_states)
    actions.extend(ep_actions)
    dones.extend(ep_dones)

states = np.array(states, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)
dones = np.array(dones, dtype=np.float32)

save_path = save_data_dir/save_data_file_name
np.savez(save_path, states=states, actions=actions, dones=dones)
print(f"专家数据已生成: {save_path}")
print(f"共 {num_trajectorys} 条轨迹，{len(states)} 个样本，成功 {success_count}/{num_trajectorys}")
