"""Deterministic discriminator inference and collection-only reward statistics."""

import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from training.common.buffer_utils import ReplayBuffer, ResidualReplayBuffer
from training.common.rl_utils import EpisodeStatsWrapper, RewardNormalizer
from training.common.train_utils import GAILTrainer, ResidualGAILTrainer
from training.policy.discriminator_policy import Discriminator


class RewardEnv(gym.Env):
    def __init__(self, truncate=False):
        self.observation_space = gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.truncate = truncate
        self.episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        return np.array([self.episode, 0], dtype=np.float32), {}

    def step(self, action):
        return (
            np.array([self.episode, 1], dtype=np.float32), 2.0,
            not self.truncate, self.truncate, {"success": not self.truncate},
        )


class DiscriminatorInferenceTest(unittest.TestCase):
    def make_discriminator(self):
        return Discriminator(
            state_dim=2, action_dim=1, hidden_dim=[8, 8], dropout=0.8,
            optimizer=dict(lr=1e-3, betas=[0.9, 0.999], eps=1e-5, weight_decay=0),
            lr_scheduler=dict(name="constant", num_warmup_steps=0),
            num_training_steps=10,
        ).to("cpu")

    def test_inference_is_deterministic_and_preserves_training_mode(self):
        discriminator = self.make_discriminator()
        states = np.ones((8, 2), dtype=np.float32)
        actions = np.ones((8, 1), dtype=np.float32)
        for training in (True, False):
            for name in ("predict_rewards", "predict_policy_prob"):
                with self.subTest(training=training, method=name):
                    discriminator.disc.train(training)
                    buffers = {
                        key: value.clone() for key, value in discriminator.disc.named_buffers()
                    }
                    rng = torch.get_rng_state().clone()
                    predict = getattr(discriminator, name)
                    first = predict(states, actions, to_numpy=False)
                    second = predict(states, actions, to_numpy=False)
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                    self.assertFalse(first.requires_grad)
                    self.assertEqual(discriminator.disc.training, training)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    for key, value in discriminator.disc.named_buffers():
                        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
                    np.testing.assert_array_equal(predict(states, actions), first.numpy())

    def test_inference_restores_mode_on_failure(self):
        discriminator = self.make_discriminator()
        for training in (True, False):
            for name in ("predict_rewards", "predict_policy_prob"):
                with self.subTest(training=training, method=name):
                    discriminator.disc.train(training)
                    with patch.object(discriminator.disc, "forward", side_effect=RuntimeError("failure")):
                        with self.assertRaisesRegex(RuntimeError, "failure"):
                            getattr(discriminator, name)(np.ones((2, 2)), np.ones((2, 1)))
                    self.assertEqual(discriminator.disc.training, training)

    def test_training_dropout_is_preserved_after_inference(self):
        discriminator = self.make_discriminator()
        states, actions = np.ones((8, 2)), np.ones((8, 1))
        discriminator.predict_rewards(states, actions)
        modes = []
        hook = discriminator.disc.register_forward_pre_hook(
            lambda module, inputs: modes.append((module.training, torch.is_grad_enabled()))
        )
        try:
            discriminator.update(states, actions, states, actions)
        finally:
            hook.remove()
        self.assertEqual(modes, [(True, True), (True, True)])


class RewardNormalizerTest(unittest.TestCase):
    def test_scales_immediate_rewards_without_centering(self):
        normalizer = RewardNormalizer()
        normalizer.update([1, 3])
        normalizer.update([5, 7])
        values = np.array([1, 3, 5, 7], dtype=np.float64)
        count = 4.0001
        mean = values.sum() / count
        expected_var = (np.square(values).sum() + 0.0001) / count - mean ** 2
        self.assertAlmostEqual(normalizer.running_ms.mean[0], mean)
        self.assertAlmostEqual(normalizer.running_ms.var[0], expected_var)
        np.testing.assert_allclose(
            normalizer.normalize(values), values / np.sqrt(expected_var + 1e-8),
        )
        self.assertGreater(normalizer.normalize(values).mean(), 0)

    def test_clips_constant_rewards_and_does_not_update_on_normalize(self):
        normalizer = RewardNormalizer(clip=2)
        normalizer.update(np.ones(1000))
        before = normalizer.running_ms.state_dict()
        np.testing.assert_array_equal(normalizer.normalize([1, -1, 0]), [2, -2, 0])
        self.assertEqual(normalizer.running_ms.state_dict(), before)

    def test_invalid_clip_is_rejected(self):
        for clip in (0, -1, float("inf"), float("nan")):
            with self.subTest(clip=clip), self.assertRaises(ValueError):
                RewardNormalizer(clip=clip)


class GAILCollectionNormalizationTest(unittest.TestCase):
    def make_trainer(self, residual=False, enabled=True, policy_updates=1):
        env = SyncVectorEnv([
            lambda: EpisodeStatsWrapper(RewardEnv()),
            lambda: EpisodeStatsWrapper(RewardEnv(truncate=True)),
        ], autoreset_mode=AutoresetMode.SAME_STEP)
        self.addCleanup(env.close)
        agent = Mock()
        agent.predict_action.side_effect = lambda obs: np.full((len(obs), 1), 0.5, dtype=np.float32)
        agent.update.return_value = {}
        discriminator = Mock()
        discriminator.predict_rewards.side_effect = (
            lambda states, actions, to_numpy: states[:, 0] + actions[:, 0]
        )
        discriminator.update.return_value = {"loss": 0.5}
        expert, generator = Mock(), Mock()
        expert.sample.return_value = dict(obs=np.zeros((2, 2)), actions=np.zeros((2, 1)))
        generator.sample.return_value = (np.zeros((2, 2)), np.zeros((2, 1)))
        kwargs = dict(
            discriminator=discriminator, expert_buffer=expert, generator_buffer=generator,
            gail_reward_scale=enabled, gail_reward_clip=5, gail_reward_coef=2,
            env_reward_weight=0.5, success_reward=10,
            total_timesteps=4, batch_size=2, learning_starts=0,
            policy_updates=policy_updates, is_draw=False, is_save_model=False,
        )
        if residual:
            base = Mock(device="cpu", prev_naction=None)
            base.sample.side_effect = lambda obs, steps: torch.full((len(obs), 1, 1), 0.25)
            trainer = ResidualGAILTrainer(
                env, object(), base, agent, ResidualReplayBuffer(32),
                residual_scale=0.1, obs_horizon=1, action_interval=1, sampling_steps=1,
                **kwargs,
            )
        else:
            trainer = GAILTrainer(env, object(), agent, ReplayBuffer(32), **kwargs)
        return trainer

    def test_collection_counts_once_and_replay_never_updates_stats(self):
        for residual in (False, True):
            with self.subTest(residual=residual):
                trainer = self.make_trainer(residual=residual, policy_updates=3)
                trainer.collector.reset(seed=1)
                initial_obs = trainer.collector.obs.copy()
                episodes = trainer.collector.step(warmup=False)
                self.assertEqual(len(episodes), 2)
                self.assertAlmostEqual(trainer.reward_normalizer.running_ms.count, 2.0001)
                call = trainer.discriminator.predict_rewards.call_args
                np.testing.assert_array_equal(call.args[0], initial_obs)
                expected_action = 0.3 if residual else 0.5
                np.testing.assert_allclose(call.args[1], expected_action)
                before = trainer.reward_normalizer.running_ms.state_dict()
                trainer._update(log_info=False)
                self.assertEqual(trainer.reward_normalizer.running_ms.state_dict(), before)
                trainer.collector.step(warmup=False)
                self.assertAlmostEqual(trainer.reward_normalizer.running_ms.count, 4.0001)
                np.testing.assert_array_equal(trainer.replay_buffer.dones[:2], [1, 0])

    def test_reward_composition_and_disabled_normalization(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                trainer = self.make_trainer(enabled=enabled)
                trainer.collector.reset(seed=1)
                trainer.collector.step(warmup=False)
                if enabled:
                    self.assertEqual(trainer.discriminator.predict_rewards.call_count, 1)
                else:
                    self.assertEqual(trainer.discriminator.predict_rewards.call_count, 0)
                batch = dict(
                    obs=np.array([[1, 0], [2, 0]], dtype=np.float32),
                    actions=np.full((2, 1), 0.5), next_obs=np.zeros((2, 2)),
                    rewards=np.array([2, 4]), successes=np.array([1, 0]),
                )
                result = trainer._prepare_batch(batch)
                raw = np.array([1.5, 2.5])
                expected = trainer.reward_normalizer.normalize(raw) if enabled else raw
                np.testing.assert_allclose(result["rewards"].ravel(), 2 * expected + [11, 2])

    def test_logs_only_gail_reward_mean_from_last_policy_batch(self):
        for residual in (False, True):
            for enabled in (False, True):
                with self.subTest(residual=residual, enabled=enabled):
                    trainer = self.make_trainer(
                        residual=residual, enabled=enabled, policy_updates=3,
                    )
                    trainer.collector.reset(seed=1)
                    trainer.collector.step(warmup=False)
                    trainer.agent.update.side_effect = lambda *args, **kwargs: {}
                    trainer.discriminator.predict_rewards.side_effect = [
                        np.array([1.0, 2.0]),
                        np.array([2.0, 3.0]),
                        np.array([3.0, 4.0]),
                    ]
                    info = trainer._update(log_info=True)
                    raw = np.array([3.0, 4.0])
                    expected = trainer.reward_normalizer.normalize(raw) if enabled else raw
                    self.assertAlmostEqual(info["gail_reward_mean"], float(np.mean(2 * expected)))
                    batch = trainer.agent.update.call_args.args[0]
                    self.assertFalse(np.isclose(info["gail_reward_mean"], batch["rewards"].mean()))
                    self.assertEqual(trainer.discriminator.predict_rewards.call_count, 3 + int(enabled))
                    trainer.wb_run = Mock()
                    logged = []
                    trainer.wb_run.log.side_effect = lambda values, step: logged.append(dict(values))
                    trainer._log({f"losses/{key}": value for key, value in info.items()})
                    trainer._flush_logs()
                    self.assertEqual(logged[0]["losses/gail_reward_mean"], info["gail_reward_mean"])
                    trainer.discriminator.predict_rewards.side_effect = None
                    trainer.discriminator.predict_rewards.return_value = raw
                    self.assertNotIn("gail_reward_mean", trainer._update(log_info=False))

    def test_training_updates_statistics_during_warmup(self):
        trainer = self.make_trainer(policy_updates=3)
        trainer.learning_starts = 2
        with patch("training.common.train_utils.tqdm"):
            trainer.train()
        self.assertAlmostEqual(trainer.reward_normalizer.running_ms.count, 4.0001)
        self.assertEqual(trainer.agent.update.call_count, 3)


if __name__ == "__main__":
    unittest.main()
