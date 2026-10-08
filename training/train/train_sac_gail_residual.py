"""Hydra 配置的 MuJoCo SAC-GAIL Residual 训练入口。"""

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
from training.common.train_utils import ResidualGAILTrainer
from training.policy.sac_policy import ResidualSACPolicy
from training.policy.discriminator_policy import Discriminator
from training.policy.diffusion_policy import DiffusionPolicy


OmegaConf.register_new_resolver("eval", eval, replace=True)

DATA_PATH = ROOT_DIR / "datasets" / "recorded_expert_data.npz"
XML_PATH = ROOT_DIR / "assets/mjcf/ur5e_assemble_sence.xml"
URDF_PATH = ROOT_DIR / "assets/urdf/ur5e_assemble.urdf"
BASE_POLICY_DIR = ROOT_DIR / "models" / "bc_model"
MODE_DIR = ROOT_DIR / "models" / "sac_gail_residual_model"
LOG_DIR = ROOT_DIR / "logs" / "sac_gail_residual_log"

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="train_sac_gail_residual")
def main(cfg: DictConfig):
    """训练、保存并评估残差 SAC-GAIL 策略。"""
    seed = int(cfg.training.seed)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 基座的预处理与权重共同冻结，不能用在线数据重新拟合。
    # 基座不参与更新，优化器与调度器仅满足构造签名，num_training_steps 传 0。
    base_policy: DiffusionPolicy = hydra.utils.instantiate(cfg.base_policy, num_training_steps=0).to(device)
    base_policy.load_model(BASE_POLICY_DIR)
    if base_policy.obs_normalizer is None:
        raise ValueError(
            f"基座模型缺少观测归一化统计量：{BASE_POLICY_DIR}。"
            "请使用 train_base_policy 保存的权重与 obs_normalizer 配套 checkpoint。"
        )
    base_policy.requires_grad_(False)
    base_policy.eval()

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

        # 残差策略与判别器创建
        residual_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=env.single_action_space.dtype,
        )
        # 梯度更新次数与 OffPolicyTrainer.train 的更新条件（global_step > learning_starts）一致
        update_steps = (
            math.ceil(int(cfg.trainer.total_timesteps) / n_envs)
            - int(cfg.trainer.learning_starts) // n_envs
        )
        agent: ResidualSACPolicy = hydra.utils.instantiate(
            cfg.policy,
            state_dim=state_dim + action_dim,
            action_dim=action_dim,
            action_space=residual_space,
            num_training_steps=update_steps * int(cfg.trainer.policy_updates),
        ).to(device)
        disc: Discriminator = hydra.utils.instantiate(
            cfg.discriminator,
            state_dim=state_dim,
            action_dim=action_dim,
            num_training_steps=update_steps * int(cfg.trainer.disc_updates),
        ).to(device)

        agent.set_obs_normalizer(base_policy.obs_normalizer)
        disc.set_obs_normalizer(base_policy.obs_normalizer)

        # 数据缓冲区构建
        expert_manager = hydra.utils.instantiate(cfg.buffers.expert)
        expert_manager.load_data(DATA_PATH, raw_obs_dim=state_dim)
        expert_buffer = expert_manager.buffer
        replay = hydra.utils.instantiate(cfg.buffers.replay)
        generator = hydra.utils.instantiate(cfg.buffers.generator, buffer_r=replay)

        # 训练器创建
        trainer: ResidualGAILTrainer = hydra.utils.instantiate(
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
