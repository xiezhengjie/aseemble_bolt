"""Hydra 配置的 MuJoCo SAC-GAIL Residual 训练入口。"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import hydra
import torch
import wandb
from gymnasium.vector import AutoresetMode
from omegaconf import DictConfig, OmegaConf
from training.common.rl_utils import make_env, set_seed
from training.policy.discriminator_policy import Discriminator
from training.policy.diffusion_policy import DiffusionPolicy
from training.policy.sac_policy import SACPolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

DATA_PATH = ROOT_DIR / "datasets" / "recorded_expert_data.npz"
XML_PATH = ROOT_DIR / "assets/mjcf/ur5e_assemble_sence.xml"
URDF_PATH = ROOT_DIR / "assets/urdf/ur5e_assemble.urdf"
BASE_POLICY_DIR = ROOT_DIR / "models" / "bc_model"
MODE_DIR = ROOT_DIR / "models" / "sac_gail_residual_model"
LOG_DIR = ROOT_DIR / "logs" / "sac_gail_residual_log"


@hydra.main(version_base=None, config_path="../../config", config_name="train_sac_gail_residual")
def main(cfg: DictConfig):
    """训练、保存并评估残差 SAC-GAIL 策略。"""
    seed = int(cfg.training.seed)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 策略日志配置
    wb_run: wandb.Run = wandb.init(
        dir=LOG_DIR,
        config=OmegaConf.to_container(cfg),
        **cfg.logging,
    )

    # 环境构建
    n_envs = int(cfg.env.num_envs)
    max_episode_steps = int(cfg.env.max_episode_steps)
    env = gym.vector.AsyncVectorEnv(
        [make_env(str(XML_PATH), str(URDF_PATH), max_episode_steps) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    eval_env = make_env(str(XML_PATH), str(URDF_PATH), max_episode_steps)()
    env.single_action_space.seed(seed)
    env.single_observation_space.seed(seed)
    state_dim = int(env.single_observation_space.shape[0])
    action_dim = int(env.single_action_space.shape[0])

    # 基座策略（冻结，仅加载权重）
    base_policy: DiffusionPolicy = hydra.utils.instantiate(cfg.base_policy).to(device)
    base_policy.load_model(BASE_POLICY_DIR)

    # 残差策略与判别器创建
    residual_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=env.single_action_space.dtype,
    )
    agent: SACPolicy = hydra.utils.instantiate(
        cfg.policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=residual_space,
    ).to(device)
    disc: Discriminator = hydra.utils.instantiate(
        cfg.discriminator,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)

    # 数据缓冲区构建
    expert_manager = hydra.utils.instantiate(cfg.buffers.expert)
    expert_manager.load_data(DATA_PATH, raw_obs_dim=state_dim)
    expert_buffer = expert_manager.buffer
    replay = hydra.utils.instantiate(cfg.buffers.replay)
    generator = hydra.utils.instantiate(cfg.buffers.generator, buffer_r=replay)

    # 训练器创建
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        env=env,
        eval_env=eval_env,
        base_agent=base_policy,
        res_agent=agent,
        replay_buffer=replay,
        discriminator=disc,
        expert_buffer=expert_buffer,
        generator_buffer=generator,
        wb_run=wb_run,
        save_model_dir=MODE_DIR,
    )

    # 训练
    try:
        trainer.train()
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
