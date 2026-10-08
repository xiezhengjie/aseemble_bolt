"""Hydra 配置的连续动作 SAC 训练入口。"""

import math
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
from training.common.train_utils import OffPolicyTrainer

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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
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
    eval_env.reset(seed=int(cfg.training.eval_env_seed))
    env.single_action_space.seed(seed)
    env.single_observation_space.seed(seed)

    # 策略创建
    state_dim = int(env.single_observation_space.shape[0])
    action_dim = int(env.single_action_space.shape[0])
    # 梯度更新次数与 OffPolicyTrainer.train 的更新条件（global_step > learning_starts）一致
    num_training_steps = (
        math.ceil(int(cfg.trainer.total_timesteps) / n_envs)
        - int(cfg.trainer.learning_starts) // n_envs
    ) * int(cfg.trainer.policy_updates)
    
    agent: SACPolicy = hydra.utils.instantiate(
        cfg.policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.single_action_space,
        num_training_steps=num_training_steps,
    ).to(device)

    # 训练器创建
    trainer: OffPolicyTrainer = hydra.utils.instantiate(
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

        # 终验
        eval_seed = int(cfg.training.eval_seed)
        set_seed(eval_seed)
        if trainer.is_save_model:
            agent.load_model(MODE_DIR / "final_model")
        metrics = trainer.evaluator.evaluate(
            eval_env,
            n_episodes=trainer.final_eval_episodes,
            seed_offset=int(cfg.training.eval_env_seed),
        )
        trainer.record_final_evaluation(metrics)
    finally:
        env.close()
        eval_env.close()
        wb_run.finish()

if __name__ == "__main__":
    main()
