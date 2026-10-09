"""模型外的观测归一化与统计量 checkpoint。"""

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

from training.common.checkpoint import has_weight, load_state_dict, save_state_dict
from training.common.rl_utils import RunningMeanStd


class ObsNormalizeWrapper(gym.Wrapper):
    """归一化物理观测，再追加吸收位；默认冻结统计，None 为恒等变换。"""

    def __init__(self, env, normalizer=None, update_stats=False, absorbing_state=False):
        # Gymnasium 的两种包装器分别要求 Env 和 VectorEnv。
        if isinstance(env, gym.vector.VectorEnv):
            gym.vector.VectorWrapper.__init__(self, env)
            space = env.single_observation_space
        else:
            gym.Wrapper.__init__(self, env)
            space = env.observation_space

        self.normalizer      = normalizer
        self.update_stats    = bool(update_stats)
        self.absorbing_state = bool(absorbing_state)

        if normalizer is not None:
            low  = np.full(space.shape, -normalizer.clip, dtype=np.float32)
            high = np.full(space.shape,  normalizer.clip, dtype=np.float32)
        else:
            low  = np.asarray(space.low,  dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
        if self.absorbing_state:
            low  = np.concatenate([np.minimum(low,  0), np.zeros(1, dtype=np.float32)])
            high = np.concatenate([np.maximum(high, 0), np.ones(1,  dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if self.normalizer is not None:
            observation = self.normalizer.normalize(observation)
        else:
            observation = observation.copy()
        if self.absorbing_state:
            observation = np.concatenate([
                observation,
                np.zeros(observation.shape[:-1] + (1,), dtype=np.float32),
            ], axis=-1)
        return observation

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if self.normalizer is not None and self.update_stats:
            self.normalizer.update(observation)
        observation = self.observation(observation)
        return observation, info

    def step(self, action):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        if self.normalizer is not None and self.update_stats:
            self.normalizer.update(observation)
        observation = self.observation(observation)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    @staticmethod
    def save_normalizer(normalizer, model_dir):
        """保存完整统计量，None 表示未启用归一化。"""
        state = None if normalizer is None else {
            **normalizer.state_dict(),
            "epsilon": normalizer.epsilon,
            "clip": normalizer.clip,
        }
        return save_state_dict(state, model_dir, "obs_normalizer")

    @staticmethod
    def load_normalizer(model_dir, normalizer=None):
        """加载配套统计；原地恢复时保留训练与评估共享的对象引用。"""
        if not has_weight(model_dir, "obs_normalizer"):
            return None
        state = load_state_dict(model_dir, "obs_normalizer", map_location="cpu")
        if state is None:
            return None
        if normalizer is None:
            normalizer = RunningMeanStd(shape=np.asarray(state["mean"]).shape)
        normalizer.epsilon = float(state.get("epsilon", normalizer.epsilon))
        normalizer.clip    = float(state.get("clip", normalizer.clip))
        normalizer.load_state_dict(state)
        return normalizer


class ObsNormalizeVectorWrapper(ObsNormalizeWrapper, gym.vector.VectorWrapper):
    """SAME_STEP 向量归一化，终帧与重置首帧使用同一统计快照。"""

    def __init__(self, env, normalizer=None, update_stats=False, absorbing_state=False):
        ObsNormalizeWrapper.__init__(self, env, normalizer, update_stats, absorbing_state)
        self.single_observation_space = self.observation_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    @property
    def autoreset_mode(self):
        return self.env.autoreset_mode

    def reset(self, *, seed=None, options=None):
        observation, infos = self.env.reset(seed=seed, options=options)
        if self.normalizer is not None and self.update_stats:
            reset_mask = None if options is None else options.get("reset_mask")
            samples = observation if reset_mask is None else observation[reset_mask]
            self.normalizer.update(samples)
        observation = self.observation(observation)
        return observation, infos

    def step(self, actions):
        (
            observation,
            rewards,
            terminated,
            truncated,
            infos,
        ) = self.env.step(actions)

        update_stats = self.normalizer is not None and self.update_stats
        if "final_obs" in infos:
            # 统计与变换在同一区块完成，终帧与自动重置首帧各计入一次。
            ended              = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
            final_mask         = np.asarray(infos.get("_final_obs", ended), dtype=bool)
            final_indices      = np.flatnonzero(final_mask)
            final_observations = infos["final_obs"]
            if update_stats:
                samples = np.asarray(observation)
                if final_indices.size:
                    final_samples = np.stack([final_observations[i] for i in final_indices])
                    samples = np.concatenate([samples, final_samples], axis=0)
                self.normalizer.update(samples)

            # object 数组保留非终止占位；普通数组扩展后逐项写入有效终帧。
            if isinstance(final_observations, np.ndarray):
                if final_observations.dtype == object:
                    transformed_finals = final_observations.copy()
                elif self.absorbing_state:
                    transformed_finals = np.concatenate([
                        np.asarray(final_observations, dtype=np.float32),
                        np.zeros(final_observations.shape[:-1] + (1,), dtype=np.float32),
                    ], axis=-1)
                else:
                    transformed_finals = np.array(final_observations, dtype=np.float32, copy=True)
            else:
                transformed_finals = list(final_observations)
            for index in final_indices:
                transformed_finals[index] = self.observation(final_observations[index])
            infos = dict(infos)
            infos["final_obs"] = transformed_finals
        elif update_stats:
            self.normalizer.update(observation)

        observation = self.observation(observation)

        return (
            observation,
            rewards,
            terminated,
            truncated,
            infos,
        )
