"""Hydra 配置的连续动作 SAC 训练入口。"""

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
from training.policy.sac_policy import SACPolicy
from training.common.rl_utils import make_env, set_seed

OmegaConf.register_new_resolver("eval", eval, replace=True)

XML_PATH = ROOT_DIR / "assets/mjcf/ur5e_assemble_sence.xml"
URDF_PATH = ROOT_DIR / "assets/urdf/ur5e_assemble.urdf"
MODE_DIR = ROOT_DIR / "models" / "sac_model"
LOG_DIR = ROOT_DIR / "logs"


@hydra.main(version_base=None, config_path="../../config", config_name="train_sac")
def main(cfg: DictConfig):
    """训练、保存并评估 SAC 策略。"""
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

    # 策略创建
    state_dim = int(env.single_observation_space.shape[0])
    action_dim = int(env.single_action_space.shape[0])
    agent: SACPolicy = hydra.utils.instantiate(
        cfg.policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.single_action_space,
    ).to(device)

    # 训练器创建
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        env=env,
        eval_env=eval_env,
        agent=agent,
        replay_buffer=hydra.utils.instantiate(cfg.buffers.replay),
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
