"""无需 MuJoCo 的采集与训练回归测试。"""

import unittest
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from training.common.eval_utils import ResidualEvaluator
from training.common.buffer_utils import ExpertBuffer, GenWindowView, ReplayBuffer, ResidualReplayBuffer
from training.common.rl_utils import EpisodeStatsWrapper
from training.common.train_utils import DataCollector, GAILTrainer, ResidualDataCollector, ResidualGAILTrainer
from training.policy.discriminator_policy import Discriminator


class CounterEnv(gym.Env):
    def __init__(self, horizon, truncate=False):
        self.observation_space = gym.spaces.Box(0, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.horizon = horizon
        self.truncate = truncate
        self.episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        return np.array([self.episode, self.steps], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        ended = self.steps == self.horizon
        obs = np.array([self.episode, self.steps], dtype=np.float32)
        return obs, float(self.steps), ended and not self.truncate, ended and self.truncate, {
            "success": ended and not self.truncate,
        }


class ChunkPolicy(torch.nn.Module):
    def __init__(self, horizon=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.horizon = horizon
        self.windows = []
        self.grad_enabled = []

    def sample(self, obs, sampling_steps):
        self.windows.append(obs.cpu().numpy().copy())
        self.grad_enabled.append(torch.is_grad_enabled())
        offsets = torch.arange(self.horizon, device=obs.device) * self.weight
        start = obs[:, -1, :1] * self.weight + obs[:, -1, 1:] * 0.01
        return (start + offsets).unsqueeze(-1)


class TrainUtilsTest(unittest.TestCase):
    def setUp(self):
        self.env = SyncVectorEnv([
            lambda: EpisodeStatsWrapper(CounterEnv(2, truncate=True)),
            lambda: EpisodeStatsWrapper(CounterEnv(4)),
        ], autoreset_mode=AutoresetMode.SAME_STEP)
        self.addCleanup(self.env.close)
        self.base = ChunkPolicy()
        self.agent = Mock()
        self.agent.predict_action.side_effect = lambda obs: np.ones((len(obs), 1), dtype=np.float32)
        self.agent.update.return_value = {"critic_loss": 0.0}
        self.replay = ResidualReplayBuffer(64)

    def collector(self, **kwargs):
        config = dict(residual_scale=0.5, obs_horizon=3, action_interval=3, sampling_steps=2)
        config.update(kwargs)
        collector = ResidualDataCollector(self.env, self.base, self.agent, self.replay, **config)
        collector.reset(seed=1)
        return collector

    def trainer(self, **kwargs):
        expert = ExpertBuffer(8)
        expert.add_batch(np.zeros((4, 2)), np.zeros((4, 1)), np.zeros(4))
        config = dict(
            residual_scale=0.5, obs_horizon=3, action_interval=3, sampling_steps=2,
            discriminator=Discriminator(2, 1, [8], lr=1e-3),
            expert_buffer=expert, generator_buffer=GenWindowView(self.replay, 8),
            total_timesteps=8, learning_starts=2, batch_size=4, disc_batch_size=4,
            is_draw=False, is_save_model=False,
        )
        config.update(kwargs)
        return ResidualGAILTrainer(self.env, object(), self.base, self.agent, self.replay, **config)

    def test_plain_replay_preserves_final_obs_and_bootstrap_mask(self):
        replay = ReplayBuffer(16)
        collector = DataCollector(self.env, self.agent, replay)
        collector.reset(seed=1)
        episodes = []
        for _ in range(4):
            episodes.extend(collector.step())
        np.testing.assert_array_equal(replay.next_obs[2], [1, 2])
        np.testing.assert_array_equal(replay.next_obs[7], [1, 4])
        np.testing.assert_array_equal(replay.obs[4], [2, 0])
        np.testing.assert_array_equal(replay.rewards[:8], [1, 1, 2, 2, 1, 3, 2, 4])
        np.testing.assert_array_equal(replay.dones[:8], [0, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(replay.successes[:8], replay.dones[:8])
        self.assertEqual([ep["episodic_return"] for ep in episodes], [3, 3, 10])

    def test_chunks_and_histories_reset_per_environment(self):
        collector = self.collector()
        for _ in range(4):
            collector.step(warmup=True)
        np.testing.assert_allclose(self.replay.actions[:8, 0], [.1, .1, .2, .2, .2, .3, .3, .13])
        self.assertEqual([len(window) for window in self.base.windows], [2, 1, 1])
        np.testing.assert_array_equal(self.base.windows[1][0], [[2, 0]] * 3)
        np.testing.assert_array_equal(self.base.windows[2][0], [[1, 1], [1, 2], [1, 3]])
        np.testing.assert_array_equal(self.replay.next_obs[2], [1, 2])
        np.testing.assert_array_equal(self.replay.next_obs[7], [1, 4])
        self.assertFalse(any(self.base.grad_enabled))
        self.assertFalse(self.base.training)
        self.agent.predict_action.assert_not_called()
        self.agent.reset_noise.assert_not_called()

    def test_warmup_and_residual_actions_have_distinct_replay_fields(self):
        collector = self.collector(residual_scale=2.0)
        collector.step(warmup=True)
        collector.step()
        np.testing.assert_array_equal(self.replay.res_actions[:2], 0)
        np.testing.assert_array_equal(self.replay.res_actions[2:4], 1)
        np.testing.assert_allclose(self.replay.obs[2:4], [[1, 1, .2]] * 2)
        np.testing.assert_array_equal(self.replay.actions[2:4], 1)
        np.testing.assert_array_equal(self.agent.predict_action.call_args.args[0], [[1, 1]] * 2)
        self.agent.reset_noise.assert_called_once_with(2)
        states, actions = GenWindowView(self.replay, 2).sample(4)
        np.testing.assert_array_equal(states, [[1, 1]] * 4)
        np.testing.assert_array_equal(actions, 1)

    def test_short_chunks_and_explicit_reset(self):
        self.base = ChunkPolicy(horizon=1)
        collector = self.collector(obs_horizon=1)
        collector.step(warmup=True)
        collector.step(warmup=True)
        np.testing.assert_allclose(self.replay.actions[:4, 0], [.1, .1, .11, .11])
        self.assertEqual([window.shape for window in self.base.windows], [(2, 1, 2)] * 2)
        collector.reset(seed=2)
        collector.step(warmup=True)
        np.testing.assert_array_equal(self.base.windows[-1][:, 0], [[3, 0], [2, 0]])
        np.testing.assert_array_equal(collector.episode_lengths, [1, 1])

    def test_invalid_chunk_settings_fail_early(self):
        for config in (dict(obs_horizon=0), dict(action_interval=0), dict(sampling_steps=0),
                       dict(residual_scale=-1), dict(residual_scale=float("nan"))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.collector(**config)
        self.base = ChunkPolicy(horizon=0)
        with self.assertRaisesRegex(ValueError, "action_horizon"):
            self.collector().step(warmup=True)

    def test_rewards_use_raw_observations_and_add_success_once(self):
        trainer = self.trainer(
            env_reward_weight=0.5, gail_reward_coef=2.0,
            gail_reward_scale=False, success_reward=10,
        )
        trainer.discriminator = Mock()
        trainer.discriminator.predict_rewards.return_value = np.array([1, 2], dtype=np.float32)
        batch = dict(obs=np.array([[1, 2, .3], [4, 5, .6]], dtype=np.float32),
                     next_obs=np.array([[1, 3], [4, 6]], dtype=np.float32),
                     actions=np.array([[.7], [.8]], dtype=np.float32),
                     res_actions=np.ones((2, 1), dtype=np.float32),
                     rewards=np.array([2, 4], dtype=np.float32),
                     dones=np.array([0, 1]), successes=np.array([0, 1]))
        prepared = trainer._prepare_batch(batch)
        np.testing.assert_array_equal(prepared["rewards"], [[3], [16]])
        np.testing.assert_array_equal(batch["rewards"], [2, 4])
        np.testing.assert_array_equal(prepared["states"], batch["obs"][..., :2])
        self.assertIs(prepared["next_states"], batch["next_obs"])
        states, actions = trainer.discriminator.predict_rewards.call_args.args
        np.testing.assert_array_equal(states, [[1, 2], [4, 5]])
        np.testing.assert_array_equal(actions, batch["actions"])
        self.assertIsInstance(trainer.evaluator, ResidualEvaluator)
        self.assertIs(trainer.evaluator.agent, self.base)
        self.assertIs(trainer.evaluator.residual_agent, self.agent)

    def test_training_loop_with_real_buffers_and_discriminator(self):
        evaluator = Mock()
        evaluator.evaluate.return_value = dict(success_rate=1.0, return_mean=3.0,
                                               return_std=0.0, peak_force_mean=0.0)
        trainer = self.trainer(evaluator=evaluator, eval_interval=4, log_interval=2)
        trainer.discriminator.update = Mock(wraps=trainer.discriminator.update)
        returns = trainer.train()
        self.assertEqual(trainer.global_step, 8)
        self.assertEqual(self.replay.size(), 8)
        self.assertEqual(self.agent.update.call_count, 3)
        self.assertEqual(trainer.discriminator.update.call_count, 6)
        self.assertEqual(evaluator.evaluate.call_count, 3)
        np.testing.assert_array_equal(returns, [3, 3, 10])
        self.assertTrue(np.isfinite(trainer.disc_losses).all())
        self.agent.set_lr_scale.assert_called_with(0.3)

    def test_plain_gail_uses_expert_dictionary_batches(self):
        replay = ReplayBuffer(16)
        expert = ExpertBuffer(8)
        expert.add_batch(np.zeros((4, 2)), np.zeros((4, 1)), np.zeros(4))
        trainer = GAILTrainer(
            self.env, object(), self.agent, replay, 8,
            discriminator=Discriminator(2, 1, [8], lr=1e-3),
            expert_buffer=expert, generator_buffer=GenWindowView(replay, 8),
            batch_size=4, disc_batch_size=4, is_draw=False, is_save_model=False,
        )
        trainer.collector.reset(seed=1)
        trainer.collector.step(warmup=True)
        info = trainer._update(log_info=True)
        self.assertTrue(np.isfinite(info["disc_loss"]))
        batch = self.agent.update.call_args.args[0]
        self.assertEqual(batch["states"].shape, (4, 2))
        self.assertEqual(batch["next_states"].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
