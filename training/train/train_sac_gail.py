"""Hydra 配置的 MuJoCo SAC-GAIL 训练入口。"""

import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import hydra
import torch
import wandb
import logging
import gymnasium as gym
from gymnasium.vector import AutoresetMode
from omegaconf import DictConfig, OmegaConf
from training.common.rl_utils import make_env, set_seed
from training.policy.discriminator_policy import Discriminator
from training.policy.sac_policy import SACPolicy
from training.common.train_utils import GAILTrainer

OmegaConf.register_new_resolver("eval", eval, replace=True)

DATA_PATH = ROOT_DIR / "datasets" / "expert_data_combined.npz"
XML_PATH = ROOT_DIR / "assets/mjcf/ur5e_assemble_sence.xml"
URDF_PATH = ROOT_DIR / "assets/urdf/ur5e_assemble.urdf"
MODE_DIR = ROOT_DIR / "models" / "sac_gail_model"
LOG_DIR = ROOT_DIR / "logs" / "sac_gail_log"

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="train_sac_gail")
def main(cfg: DictConfig):
    """训练、保存并评估 SAC-GAIL 策略。"""
    seed = int(cfg.training.seed)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with ExitStack() as stack:
        # 策略日志配置
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        wb_run: wandb.Run = wandb.init(
            dir=LOG_DIR,
            config=OmegaConf.to_container(cfg),
            **cfg.logging,
        )
        stack.callback(wb_run.finish)

        # 环境构建
        n_envs = int(cfg.env.num_envs)
        max_episode_steps = int(cfg.env.max_episode_steps)
        env = gym.vector.AsyncVectorEnv(
            [make_env(str(XML_PATH), str(URDF_PATH), max_episode_steps) for _ in range(n_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        stack.callback(env.close)
        eval_episodes = max(
            int(cfg.trainer.eval_episodes), int(cfg.trainer.final_eval_episodes),
        )
        eval_num_envs = int(cfg.training.eval_num_envs)
        if eval_num_envs <= 0:
            raise ValueError("training.eval_num_envs 必须为正数")
        eval_num_envs = min(eval_num_envs, eval_episodes)
        eval_env = gym.vector.AsyncVectorEnv(
            [make_env(str(XML_PATH), str(URDF_PATH), max_episode_steps)
             for _ in range(eval_num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        stack.callback(eval_env.close)
        env.single_action_space.seed(seed)
        env.single_observation_space.seed(seed)
        state_dim = int(env.single_observation_space.shape[0])
        action_dim = int(env.single_action_space.shape[0])

        # 策略与判别器创建
        # 梯度更新次数与 OffPolicyTrainer.train 的更新条件（global_step > learning_starts）一致
        update_steps = (
            math.ceil(int(cfg.trainer.total_timesteps) / n_envs)
            - int(cfg.trainer.learning_starts) // n_envs
        )
        agent: SACPolicy = hydra.utils.instantiate(
            cfg.policy,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=env.single_action_space,
            num_training_steps=update_steps * int(cfg.trainer.policy_updates),
        ).to(device)
        disc: Discriminator = hydra.utils.instantiate(
            cfg.discriminator,
            state_dim=state_dim,
            action_dim=action_dim,
            num_training_steps=update_steps * int(cfg.trainer.disc_updates),
        ).to(device)

        # 数据缓冲区构建
        expert_manager = hydra.utils.instantiate(cfg.buffers.expert)
        expert_manager.load_data(DATA_PATH, raw_obs_dim=state_dim)
        expert_buffer = expert_manager.buffer
        replay = hydra.utils.instantiate(cfg.buffers.replay)
        generator = hydra.utils.instantiate(cfg.buffers.generator, buffer_r=replay)

        # 训练器创建
        trainer: GAILTrainer = hydra.utils.instantiate(
            cfg.trainer,
            env=env,
            eval_env=eval_env,
            agent=agent,
            replay_buffer=replay,
            discriminator=disc,
            expert_buffer=expert_buffer,
            generator_buffer=generator,
            wb_run=wb_run,
            save_model_dir=MODE_DIR,
            eval_seed=int(cfg.training.eval_seed),
            eval_env_seed=int(cfg.training.eval_env_seed),
        )

        # 训练
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


if __name__ == "__main__":
    main()
