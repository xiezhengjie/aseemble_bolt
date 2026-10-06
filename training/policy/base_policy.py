
from typing import Dict
import torch
from training.common.checkpoint import has_weight, load_state_dict, save_state_dict
from training.common.rl_utils import RunningMeanStd
from training.model.common.module_attr_mixin import ModuleAttrMixin

class BasePolicy(ModuleAttrMixin):
    def __init__(self):
        super().__init__()
        self.obs_normalizer = None

    def set_obs_normalizer(self, normalizer):
        """复制固定观测统计量；预测和更新仅应用变换，不更新统计量。"""
        if normalizer is None:
            self.obs_normalizer = None
            return
        obs_dim = getattr(self, 'obs_dim', getattr(self, 'state_dim', None))
        if obs_dim is not None and normalizer.mean.shape != (obs_dim,):
            raise ValueError(f"观测归一化器维度不匹配：期望 {(obs_dim,)}，实际 {normalizer.mean.shape}")
        self.obs_normalizer = RunningMeanStd(
            shape=normalizer.mean.shape,
            epsilon=normalizer.epsilon,
            clip=normalizer.clip,
        )
        self.obs_normalizer.load_state_dict(normalizer.state_dict())

    def normalize_obs(self, obs):
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer.normalize(obs)

    def _save_obs_normalizer(self, model_dir):
        normalizer = self.obs_normalizer
        state = None if normalizer is None else {
            **normalizer.state_dict(),
            'epsilon': normalizer.epsilon,
            'clip': normalizer.clip,
        }
        save_state_dict(state, model_dir, 'obs_normalizer')

    def _load_obs_normalizer(self, model_dir):
        self.obs_normalizer = None
        if has_weight(model_dir, 'obs_normalizer'):
            state = load_state_dict(model_dir, 'obs_normalizer', map_location='cpu')
            if state is not None:
                normalizer = RunningMeanStd(
                    shape=(len(state['mean']),),
                    epsilon=state['epsilon'],
                    clip=state['clip'],
                )
                normalizer.load_state_dict(state)
                self.set_obs_normalizer(normalizer)

    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: B,To,Do
        return: 
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass


    