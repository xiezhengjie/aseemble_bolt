"""训练 diffusion policy 的行为克隆模型。"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import numpy as np
import torch
import wandb
import logging
from omegaconf import DictConfig, OmegaConf
from training.common.buffer_utils import ExpertDataManager
from training.common.eval_utils import BaseChunkPolicyEvaluator
from training.common.rl_utils import set_seed, EpisodeStatsWrapper
from training.common.train_utils import SupervisedPolicyTrainer
from training.envs.assemble_mujoco_env import AssembleMuJoCoEnv
from training.policy.diffusion_policy import DiffusionPolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

DATA_PATH = ROOT_DIR / "datasets" / "expert_data_combined.npz"
XML_PATH = ROOT_DIR / "assets/mjcf/ur5e_assemble_sence.xml"
URDF_PATH = ROOT_DIR / "assets/urdf/ur5e_assemble.urdf"
MODE_DIR = ROOT_DIR / "models" / "bc_model"
LOG_DIR = ROOT_DIR / "logs" 

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="train_base_policy")
def main(cfg: DictConfig):
    """训练、保存并按配置执行可选评估。"""
    set_seed(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 策略日志配置
    wb_run: wandb.Run = wandb.init(
        dir=LOG_DIR,
        config=OmegaConf.to_container(cfg),
        **cfg.logging
    )
    
    # 数据加载
    manager = ExpertDataManager()
    manager.load_data(DATA_PATH, raw_obs_dim=cfg.policy.obs_dim)
    train_loader, val_loader = manager.trans_dataloader(
        split=cfg.training.split,
        batch_size=int(cfg.training.batch_size),
        device=device,
        seed=int(cfg.training.split_seed),
        horizon=int(cfg.policy.horizon),
        n_obs_steps=int(cfg.policy.n_obs_steps),
        n_action_steps=int(cfg.policy.n_action_steps),
        num_workers=int(cfg.training.num_workers),
    )

    # 策略创建
    policy: DiffusionPolicy =  hydra.utils.instantiate(cfg.policy).to(device)
    sampler = train_loader.dataset.sampler
    data = sampler.replay_buffer
    train_frames = np.zeros(len(data['obs']), dtype=bool)
    for start, end, _, _ in sampler.indices:
        train_frames[start:end] = True
    if not train_frames.any():
        raise ValueError("训练集没有可用于拟合归一化统计量的帧")
    policy.fit_obs_normalizer(data['obs'][train_frames])

    # 训练
    trainer = SupervisedPolicyTrainer(
        agent=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        total_epochs=int(cfg.training.epochs),
        model_dir=MODE_DIR,
        patience=int(cfg.training.patience),
        wb_run=wb_run,
    )
    trainer.train()

    # 验证
    set_seed(int(cfg.training.eval_seed))
    policy.load_model(MODE_DIR)
    env = AssembleMuJoCoEnv(
            xml_path=str(XML_PATH),
            urdf_path=str(URDF_PATH),
            render_mode=None,
            max_episodic_steps=400,
        )
    env = EpisodeStatsWrapper(env)
    evaluator = BaseChunkPolicyEvaluator(
        policy,
        obs_horizon=policy.n_obs_steps,
        action_interval=policy.n_action_steps,
        sampling_steps=policy.num_inference_steps,
    )
    metrics = evaluator.evaluate(
        env,
        n_episodes=int(cfg.training.eval_episodes),
        seed_offset=int(cfg.training.eval_seed),
    )
    logger.info(metrics)
    env.close()
    wb_run.finish()


if __name__ == "__main__":
    main()
