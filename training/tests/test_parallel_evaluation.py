"""Standalone unittest coverage for vector evaluation without MuJoCo simulation.

Run from the repository root:
    python3 -m unittest training.tests.test_parallel_evaluation -v
"""

from functools import partial
import unittest

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

from training.common.eval_utils import (
    BaseChunkPolicyEvaluator,
    Evaluator,
    ResidualEvaluator,
)
from training.common.rl_utils import EpisodeStatsWrapper


class ToyEpisodeEnv(gym.Env):
    """Distinct slot/step/reset observations and deliberately hostile reset info."""

    def __init__(self, slot, horizon, truncate=False):
        self.slot = slot
        self.horizon = horizon
        self.truncate = truncate
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            np.array([-1.0, -0.5], dtype=np.float32),
            np.array([0.7, 1.0], dtype=np.float32))
        self.reset_seeds = []
        self.actions = []
        self.reset_count = 0
        self.episode_seed = 0
        self.autoreset_episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self.reset_count += 1
        self.steps = 0
        if seed is not None:
            self.episode_seed = seed
            self.autoreset_episode = 0
        else:
            self.autoreset_episode += 1
        self.random_samples = getattr(self, "random_samples", [])
        self.random_samples.append(float(self.np_random.random()))
        obs = np.array([self.slot, 0, self.reset_count], dtype=np.float32)
        return obs, {
            "force": np.full(7, 1e6, dtype=np.float32),
            "success": self.truncate,
            "prev_state": -99,
            "episodic_return": -1000.0,
            "episodic_length": 999,
        }

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        ended = self.steps == self.horizon
        factor = self.slot + 1
        force = np.array([3 * factor, 4 * factor, 0, 0, 0, 0, 1e5],
                         dtype=np.float32) if ended else np.array(
                             [factor, 0, 0, 0, 0, 0, 1e5], dtype=np.float32)
        info = {
            "force": force,
            "success": not self.truncate if ended else False,
            "prev_state": self.slot + 10,
            # The real EpisodeStatsWrapper's inner metrics must take priority.
            "episodic_return": -2000.0,
            "episodic_length": 777,
        }
        reward = float(self.episode_seed + self.steps
                       + 10000 * self.autoreset_episode)
        obs = np.array([self.slot, self.steps, self.reset_count], dtype=np.float32)
        return (obs, reward, ended and not self.truncate,
                ended and self.truncate, info)


def make_toy_env(slot, horizon, truncate=False):
    return EpisodeStatsWrapper(ToyEpisodeEnv(slot, horizon, truncate))


class ActionAgent:
    def __init__(self, action=None, squeeze_single=False):
        self.action = action
        self.squeeze_single = squeeze_single
        self.calls = []

    def predict_action(self, obs, deterministic=False):
        obs = np.asarray(obs)
        self.calls.append((obs.copy(), deterministic))
        if self.action is None:
            actions = np.stack([0.1 * (obs[:, 0] + 1), -0.1 * obs[:, 1]], axis=-1)
        else:
            actions = np.tile(np.asarray(self.action, dtype=np.float32), (len(obs), 1))
        return actions[0] if self.squeeze_single and len(obs) == 1 else actions


class ChunkAgent:
    """Two-step plans with a slot-specific, in-place-updated warm-start prior."""

    device = torch.device("cpu")

    def __init__(self, plan_length=2, fixed_action=None):
        self.plan_length = plan_length
        self.fixed_action = fixed_action
        self.prev_naction = None
        self.reset_calls = 0
        self.calls = []

    def reset(self):
        self.reset_calls += 1
        self.prev_naction = None

    def sample(self, obs, sampling_steps):
        prior = None if self.prev_naction is None else self.prev_naction.clone()
        self.calls.append({
            "history": obs.cpu().numpy().copy(),
            "prior": prior,
            "sampling_steps": sampling_steps,
            "grad_enabled": torch.is_grad_enabled(),
        })
        if self.prev_naction is None:
            self.prev_naction = torch.zeros((len(obs), 2, 2), device=obs.device)
        self.prev_naction.add_(obs[:, -1, :1, None] + 1)
        if self.fixed_action is not None:
            return torch.tensor(self.fixed_action).repeat(len(obs), self.plan_length, 1)
        start = obs[:, -1, 0] * 0.1 + obs[:, -1, 1] * 0.01
        offsets = torch.arange(self.plan_length, dtype=torch.float32) * 0.02
        first = start[:, None] + offsets[None, :]
        return torch.stack([first, -first], dim=-1)


class VectorTestCase(unittest.TestCase):
    def make_vector(self, specs=((1, False), (2, True), (4, False))):
        env = SyncVectorEnv([
            partial(make_toy_env, slot, horizon, truncate)
            for slot, (horizon, truncate) in enumerate(specs)
        ], autoreset_mode=AutoresetMode.SAME_STEP)
        self.addCleanup(env.close)
        return env

    def assert_stats(self, stats, returns, lengths, successes, peaks, states):
        self.assertEqual(stats["n_episodes"], len(returns))
        self.assertAlmostEqual(stats["return_mean"], float(np.mean(returns)))
        self.assertAlmostEqual(stats["return_std"], float(np.std(returns)))
        self.assertAlmostEqual(stats["length_mean"], float(np.mean(lengths)))
        self.assertAlmostEqual(stats["success_rate"], float(np.mean(successes)))
        self.assertAlmostEqual(stats["peak_force_mean"], float(np.mean(peaks)))
        self.assertEqual(stats["state_dist"], states)


class ParallelEvaluatorTest(VectorTestCase):
    def test_different_horizons_nested_stats_terminal_force_and_first_episode_only(self):
        env = self.make_vector()
        agent = ActionAgent()
        stats = Evaluator(agent).evaluate(env, n_episodes=3, seed_offset=100)
        self.assert_stats(stats, [101, 205, 418], [1, 2, 4], [1, 0, 1],
                          [5, 10, 15], {10: 1, 11: 1, 12: 1})
        # Slot zero completes four episodes while the long slot completes one.
        self.assertEqual([len(e.unwrapped.reset_seeds) for e in env.envs], [5, 3, 2])
        self.assertEqual([len(obs) for obs, _ in agent.calls], [3, 2, 1, 1])
        self.assertTrue(all(deterministic for _, deterministic in agent.calls))
        np.testing.assert_array_equal(agent.calls[0][0],
                                      [[0, 0, 1], [1, 0, 1], [2, 0, 1]])
        np.testing.assert_array_equal(agent.calls[1][0], [[1, 1, 1], [2, 1, 1]])
        for slot, wrapped in enumerate(env.envs):
            np.testing.assert_allclose(wrapped.unwrapped.actions[0],
                                       [0.1 * (slot + 1), 0])
        np.testing.assert_array_equal(env.envs[0].unwrapped.actions[1:], 0)
        np.testing.assert_array_equal(env.envs[1].unwrapped.actions[2:], 0)

    def test_nondivisible_episode_count_and_batch_seed_coverage(self):
        env = self.make_vector()
        agent = ActionAgent(squeeze_single=True)
        stats = Evaluator(agent).evaluate(env, n_episodes=5, seed_offset=100)
        self.assert_stats(stats, [101, 205, 418, 104, 211], [1, 2, 4, 1, 2],
                          [1, 0, 1, 1, 0], [5, 10, 15, 5, 10],
                          {10: 2, 11: 2, 12: 1})
        self.assertEqual([[seed for seed in e.unwrapped.reset_seeds if seed is not None]
                          for e in env.envs], [[100, 103], [101, 104], [102, 105]])
        self.assertEqual([len(obs) for obs, _ in agent.calls], [3, 2, 1, 1, 2, 1])
        self.assertTrue(all(flag for _, flag in agent.calls))
        # All vector slots are stepped, but the unused last-batch slot gets zeros.
        np.testing.assert_array_equal(env.envs[2].unwrapped.actions[4:], 0)

    def test_fewer_episodes_than_slots(self):
        env = self.make_vector()
        agent = ActionAgent(squeeze_single=True)
        stats = Evaluator(agent).evaluate(env, n_episodes=1, seed_offset=9)
        self.assert_stats(stats, [10], [1], [1], [5], {10: 1})
        self.assertEqual([len(obs) for obs, _ in agent.calls], [1])
        self.assertEqual([e.unwrapped.reset_seeds[0] for e in env.envs], [9, 10, 11])
        for wrapped in env.envs[1:]:
            np.testing.assert_array_equal(wrapped.unwrapped.actions, 0)

    def test_seed_none_preserves_environment_rng_stream(self):
        env = self.make_vector(((1, False), (2, False)))
        env.reset(seed=[31, 32])
        Evaluator(ActionAgent()).evaluate(env, n_episodes=3, seed_offset=None)
        for wrapped, seed in zip(env.envs, [31, 32]):
            toy = wrapped.unwrapped
            self.assertTrue(all(value is None for value in toy.reset_seeds[1:]))
            np.testing.assert_array_equal(toy.random_samples,
                                          np.random.default_rng(seed).random(len(toy.random_samples)))

    def test_nested_final_info_masks_and_metric_precedence(self):
        env = self.make_vector(((1, False), (2, True)))
        env.reset(seed=[1, 2])
        _, _, terminated, truncated, infos = env.step(np.zeros((2, 2), np.float32))
        np.testing.assert_array_equal(terminated | truncated, [True, False])
        np.testing.assert_array_equal(infos["_final_info"], [True, False])
        self.assertEqual(infos["final_info"]["final_info"]["episodic_return"][0], 2)
        self.assertEqual(infos["final_info"]["prev_state"][0], 10)
        self.assertEqual(infos["force"][0, 0], 1e6)
        # Reset success differs from both outer and nested terminal success.
        self.assertFalse(infos["success"][0])
        self.assertTrue(infos["final_info"]["success"][0])
        self.assertTrue(infos["final_info"]["final_info"]["success"][0])

    def test_success_falls_back_to_nested_terminal_info(self):
        class InnerSuccessWrapper(gym.Wrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    info.pop("success")
                return obs, reward, terminated, truncated, info

        env = SyncVectorEnv([lambda: InnerSuccessWrapper(make_toy_env(0, 1))],
                            autoreset_mode=AutoresetMode.SAME_STEP)
        self.addCleanup(env.close)
        stats = Evaluator(ActionAgent()).evaluate(env, n_episodes=1, seed_offset=8)
        self.assert_stats(stats, [9], [1], [1], [5], {10: 1})

    def test_missing_terminal_success_and_force_ignore_reset_info(self):
        class MissingTerminalFieldsWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                info["success"] = True
                return obs, info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                info.pop("force", None)
                if terminated or truncated:
                    info.pop("success", None)
                    info["final_info"].pop("success", None)
                return obs, reward, terminated, truncated, info

        env = SyncVectorEnv([
            lambda: MissingTerminalFieldsWrapper(make_toy_env(0, 1)),
            lambda: MissingTerminalFieldsWrapper(make_toy_env(1, 3, True)),
        ], autoreset_mode=AutoresetMode.SAME_STEP)
        self.addCleanup(env.close)
        stats = Evaluator(ActionAgent()).evaluate(env, n_episodes=2, seed_offset=10)
        self.assert_stats(stats, [11, 39], [1, 3], [0, 0], [0, 0],
                          {10: 1, 11: 1})

    def test_invalid_episode_count_and_autoreset_mode(self):
        env = self.make_vector()
        evaluator = Evaluator(ActionAgent())
        for count in (0, -1):
            with self.subTest(n_episodes=count), self.assertRaises(ValueError):
                evaluator.evaluate(env, n_episodes=count)
        for mode in (AutoresetMode.NEXT_STEP, AutoresetMode.DISABLED):
            other = SyncVectorEnv([partial(make_toy_env, 0, 1)], autoreset_mode=mode)
            self.addCleanup(other.close)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "SAME_STEP"):
                evaluator.evaluate(other, n_episodes=1)

    def test_async_vector_smoke(self):
        # Use real worker processes; assertion or worker errors must fail the test.
        env = AsyncVectorEnv([partial(make_toy_env, 0, 1),
                              partial(make_toy_env, 1, 3, True)],
                             autoreset_mode=AutoresetMode.SAME_STEP)
        self.addCleanup(env.close)
        stats = Evaluator(ActionAgent(squeeze_single=True)).evaluate(
            env, n_episodes=3, seed_offset=20)
        self.assert_stats(stats, [21, 69, 23], [1, 3, 1], [1, 0, 1],
                          [5, 10, 5], {10: 2, 11: 1})


class BaseChunkVectorEvaluatorTest(VectorTestCase):
    def test_history_short_plans_replan_subbatches_and_prior_isolation(self):
        env = self.make_vector(((1, False), (3, False), (5, False)))
        agent = ChunkAgent()
        evaluator = BaseChunkPolicyEvaluator(agent, obs_horizon=3,
                                             action_interval=4, sampling_steps=9)
        stats = evaluator.evaluate(env, n_episodes=3, seed_offset=50)
        self.assertEqual(stats["n_episodes"], 3)
        self.assertEqual(agent.reset_calls, 1)
        self.assertEqual([len(call["history"]) for call in agent.calls], [3, 2, 1])
        self.assertTrue(all(call["sampling_steps"] == 9 for call in agent.calls))
        self.assertFalse(any(call["grad_enabled"] for call in agent.calls))
        np.testing.assert_array_equal(agent.calls[0]["history"], [
            [[0, 0, 1]] * 3, [[1, 0, 1]] * 3, [[2, 0, 1]] * 3])
        np.testing.assert_array_equal(agent.calls[1]["history"], [
            [[1, 0, 1], [1, 1, 1], [1, 2, 1]],
            [[2, 0, 1], [2, 1, 1], [2, 2, 1]]])
        np.testing.assert_array_equal(agent.calls[2]["history"], [
            [[2, 2, 1], [2, 3, 1], [2, 4, 1]]])
        self.assertIsNone(agent.calls[0]["prior"])
        torch.testing.assert_close(agent.calls[1]["prior"],
                                   torch.tensor([2., 3.])[:, None, None].expand(2, 2, 2))
        torch.testing.assert_close(agent.calls[2]["prior"], torch.full((1, 2, 2), 6.))
        self.assertIsNone(agent.prev_naction)
        torch.testing.assert_close(evaluator._vector_state["prev_naction"],
                                   torch.zeros((3, 2, 2)))
        for slot, expected in enumerate(([[0, 0]],
                                        [[.1, -.1], [.12, -.12], [.12, -.12]],
                                        [[.2, -.2], [.22, -.22], [.22, -.22],
                                         [.24, -.24], [.24, -.24]])):
            np.testing.assert_allclose(env.envs[slot].unwrapped.actions[:len(expected)],
                                       expected, atol=1e-7)

    def test_short_last_batch_resets_history_and_prior(self):
        env = self.make_vector(((1, False), (3, False), (5, False)))
        agent = ChunkAgent()
        evaluator = BaseChunkPolicyEvaluator(agent, 3, 4, 2)
        evaluator.evaluate(env, n_episodes=4, seed_offset=70)
        self.assertEqual(agent.reset_calls, 2)
        self.assertEqual([len(call["history"]) for call in agent.calls], [3, 2, 1, 1])
        last = agent.calls[-1]
        self.assertIsNone(last["prior"])
        # Slot zero auto-resets five times in batch one before explicit batch two.
        np.testing.assert_array_equal(last["history"], [[[0, 0, 7]] * 3])
        self.assertIsNone(agent.prev_naction)
        self.assertEqual([[seed for seed in e.unwrapped.reset_seeds if seed is not None]
                          for e in env.envs], [[70, 73], [71, 74], [72, 75]])

    def test_action_interval_limits_longer_plan(self):
        env = self.make_vector(((3, False),))
        agent = ChunkAgent(plan_length=5)
        BaseChunkPolicyEvaluator(agent, 2, 2, 1).evaluate(env, n_episodes=1)
        self.assertEqual(len(agent.calls), 2)
        np.testing.assert_array_equal(agent.calls[1]["history"],
                                      [[[0, 1, 1], [0, 2, 1]]])

    def test_sampling_restores_external_prior_and_clones_slot_prior(self):
        env = self.make_vector()
        agent = ChunkAgent()
        evaluator = BaseChunkPolicyEvaluator(agent, 2, 2, 1)
        obs, _ = env.reset(seed=1)
        state = evaluator._vector_reset(obs, env)
        external = torch.full((1, 2, 2), 99.)
        agent.prev_naction = external
        state["prev_naction"] = torch.tensor([10., 20., 30.])[:, None, None].repeat(1, 2, 2)
        evaluator._sample_vector_plans(state, np.array([2, 0]))
        self.assertIs(agent.prev_naction, external)
        torch.testing.assert_close(external, torch.full((1, 2, 2), 99.))
        torch.testing.assert_close(agent.calls[-1]["prior"],
                                   torch.tensor([30., 10.])[:, None, None].expand(2, 2, 2))
        torch.testing.assert_close(state["prev_naction"],
                                   torch.tensor([11., 20., 33.])[:, None, None].expand(3, 2, 2))


class ResidualVectorEvaluatorTest(VectorTestCase):
    def test_augmented_observations_deterministic_subbatches_scale_and_clip(self):
        env = self.make_vector(((1, False), (3, False)))
        base = ChunkAgent(fixed_action=[.4, -.2])
        residual = ActionAgent(action=[.5, -1.], squeeze_single=True)
        evaluator = ResidualEvaluator(base, residual, residual_scale=2.,
                                      obs_horizon=2, action_interval=3, sampling_steps=4)
        stats = evaluator.evaluate(env, n_episodes=3, seed_offset=10)
        self.assertEqual(stats["n_episodes"], 3)
        self.assertEqual([len(obs) for obs, _ in residual.calls], [2, 1, 1, 1])
        self.assertTrue(all(flag for _, flag in residual.calls))
        np.testing.assert_allclose(residual.calls[0][0],
                                   [[0, 0, 1, .4, -.2], [1, 0, 1, .4, -.2]])
        np.testing.assert_allclose(residual.calls[1][0], [[1, 1, 1, .4, -.2]])
        np.testing.assert_allclose(residual.calls[2][0], [[1, 2, 1, .4, -.2]])
        np.testing.assert_allclose(residual.calls[3][0], [[0, 0, 5, .4, -.2]])
        np.testing.assert_allclose(env.envs[0].unwrapped.actions[0], [.7, -.5])
        np.testing.assert_array_equal(env.envs[0].unwrapped.actions[1:3], 0)
        np.testing.assert_allclose(env.envs[0].unwrapped.actions[3], [.7, -.5])
        np.testing.assert_allclose(env.envs[1].unwrapped.actions[:3], [[.7, -.5]] * 3)
        np.testing.assert_array_equal(env.envs[1].unwrapped.actions[3:], 0)

    def test_residual_scale_without_clipping(self):
        env = self.make_vector(((1, False), (1, True)))
        evaluator = ResidualEvaluator(
            ChunkAgent(fixed_action=[.1, .2]), ActionAgent(action=[.2, -.4]),
            residual_scale=.5, obs_horizon=1, action_interval=1, sampling_steps=1)
        evaluator.evaluate(env, n_episodes=2)
        for wrapped in env.envs:
            np.testing.assert_allclose(wrapped.unwrapped.actions[0], [.2, 0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
