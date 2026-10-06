"""固定状态统计量、网络输入与评估接口的回归测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from training.common.eval_utils import Evaluator, ResidualEvaluator
from training.common.rl_utils import RunningMeanStd
from training.policy.base_policy import BasePolicy
from training.policy.discriminator_policy import Discriminator
from training.policy.sac_policy import SACPolicy


def make_agent():
    return SACPolicy(
        state_dim=2, hidden_dim=[8, 8], action_dim=1,
        action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3,
        tau=0.005, gamma=0.99, use_sde=False,
    ).to('cpu')


def make_normalizer():
    normalizer = RunningMeanStd((2,), epsilon=1e-3, clip=10.)
    normalizer.update(np.array([[10., 20.], [30., 60.]], dtype=np.float32))
    return normalizer


class FixedObservationNormalizationTest(unittest.TestCase):
    def test_sac_normalizes_collection_and_all_update_inputs_once(self):
        agent = make_agent()
        source = make_normalizer()
        agent.set_obs_normalizer(source)
        snapshot = agent.obs_normalizer.state_dict()
        states = np.array([[30., 60.], [10., 20.]], dtype=np.float32)
        next_states = states + 5
        actions = np.array([[.2], [-.3]], dtype=np.float32)
        batch = dict(states=states, next_states=next_states, actions=actions,
                     rewards=np.ones(2), dones=np.zeros(2))
        expected = agent.normalize_obs(torch.from_numpy(states))
        expected_next = agent.normalize_obs(torch.from_numpy(next_states))
        actor_inputs, critic_inputs, target_inputs = [], [], []
        handles = [
            agent.actor.register_forward_pre_hook(
                lambda module, args: actor_inputs.append(args[0].detach().clone())),
            agent.critic_1.register_forward_pre_hook(
                lambda module, args: critic_inputs.append(tuple(x.detach().clone() for x in args))),
            agent.target_critic_1.register_forward_pre_hook(
                lambda module, args: target_inputs.append(args[0].detach().clone())),
        ]
        try:
            with patch.object(agent.obs_normalizer, 'update', side_effect=AssertionError('online fit')):
                agent.predict_action(states, deterministic=True)
                agent.update(batch)
                agent.calc_target(torch.ones(2, 1), torch.from_numpy(next_states), torch.zeros(2, 1))
        finally:
            for handle in handles:
                handle.remove()
        for actual in actor_inputs[:2]:
            torch.testing.assert_close(actual, expected)
        for actual in actor_inputs[2:]:
            torch.testing.assert_close(actual, expected_next)
        torch.testing.assert_close(critic_inputs[0][0], expected)
        torch.testing.assert_close(critic_inputs[0][1], torch.from_numpy(actions))
        torch.testing.assert_close(critic_inputs[1][0], expected)
        for actual in target_inputs:
            torch.testing.assert_close(actual, expected_next)
        self.assertEqual(agent.obs_normalizer.state_dict(), snapshot)
        self.assertEqual(source.state_dict(), snapshot)
        np.testing.assert_array_equal(batch['states'], states)
        np.testing.assert_array_equal(batch['actions'], actions)

    def test_discriminator_normalizes_both_domains_and_reward_queries(self):
        disc = Discriminator(2, 1, [8], lr=1e-3).to('cpu')
        disc.set_obs_normalizer(make_normalizer())
        snapshot = disc.obs_normalizer.state_dict()
        expert = np.array([[10., 20.], [30., 60.]], dtype=np.float32)
        generated = expert + 5
        actions = np.array([[.7], [-.8]], dtype=np.float32)
        captured = []
        handle = disc.disc.register_forward_pre_hook(
            lambda module, args: captured.append(tuple(x.detach().clone() for x in args)))
        try:
            with patch.object(disc.obs_normalizer, 'update', side_effect=AssertionError('online fit')):
                disc.update(expert, actions, generated, actions)
                disc.predict_rewards(generated, actions)
                disc.predict_policy_prob(generated, actions)
        finally:
            handle.remove()
        torch.testing.assert_close(captured[0][0], disc.normalize_obs(torch.from_numpy(expert)))
        for state, action in captured:
            torch.testing.assert_close(action, torch.from_numpy(actions))
        for state, _ in captured[1:]:
            torch.testing.assert_close(state, disc.normalize_obs(torch.from_numpy(generated)))
        self.assertEqual(disc.obs_normalizer.state_dict(), snapshot)

    def test_normalizer_copies_are_independent_and_dimension_checked(self):
        source = make_normalizer()
        agent = make_agent()
        disc = Discriminator(2, 1, [8], lr=1e-3)
        agent.set_obs_normalizer(source)
        disc.set_obs_normalizer(source)
        self.assertIsNot(agent.obs_normalizer, source)
        self.assertIsNot(disc.obs_normalizer, source)
        source.update(np.full((4, 2), 100.))
        self.assertEqual(agent.obs_normalizer.state_dict(), disc.obs_normalizer.state_dict())
        self.assertNotEqual(agent.obs_normalizer.state_dict(), source.state_dict())
        with self.assertRaisesRegex(ValueError, '维度不匹配'):
            agent.set_obs_normalizer(RunningMeanStd((3,)))

    def test_checkpoint_restores_fixed_stats_and_prediction(self):
        obs = np.array([30., 60.], dtype=np.float32)
        for factory in (make_agent, lambda: Discriminator(2, 1, [8], lr=1e-3).to('cpu')):
            with self.subTest(factory=factory), tempfile.TemporaryDirectory() as folder:
                original = factory()
                original.set_obs_normalizer(make_normalizer())
                if isinstance(original, SACPolicy):
                    predict = lambda model: model.predict_action(obs, deterministic=True)
                else:
                    original.disc.eval()
                    predict = lambda model: model.predict_rewards(obs, np.array([.2], dtype=np.float32))
                expected = predict(original)
                original.save_model(folder)
                restored = factory()
                restored.load_model(folder)
                if isinstance(restored, Discriminator):
                    restored.disc.eval()
                self.assertEqual(restored.obs_normalizer.state_dict(), original.obs_normalizer.state_dict())
                self.assertEqual(restored.obs_normalizer.epsilon, original.obs_normalizer.epsilon)
                self.assertEqual(restored.obs_normalizer.clip, original.obs_normalizer.clip)
                np.testing.assert_allclose(predict(restored), expected)
                Path(folder, 'obs_normalizer.pt').unlink()
                restored.load_model(folder)
                self.assertIsNone(restored.obs_normalizer)


class ActionTrackingEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([10., 20.], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        return np.array([10. + self.steps, 20.], dtype=np.float32), 0., self.steps == 2, False, {
            'force': np.zeros(3), 'success': self.steps == 2,
        }


class FixedChunkPolicy(BasePolicy):
    def sample(self, obs, sampling_steps):
        return torch.full((len(obs), 2, 1), .9)


class EvaluationInterfaceTest(unittest.TestCase):
    def test_plain_evaluator_uses_current_observation_and_deterministic_api(self):
        agent = make_agent()
        agent.set_obs_normalizer(make_normalizer())
        env = ActionTrackingEnv()
        with patch.object(agent, 'predict_action', wraps=agent.predict_action) as predict:
            metrics = Evaluator(agent).evaluate(env, n_episodes=1)
        self.assertEqual(metrics['success_rate'], 1.)
        self.assertEqual(predict.call_count, 2)
        np.testing.assert_array_equal(predict.call_args_list[0].args[0], [10., 20.])
        np.testing.assert_array_equal(predict.call_args_list[1].args[0], [11., 20.])
        self.assertTrue(all(call.kwargs['deterministic'] for call in predict.call_args_list))

    def test_residual_evaluator_uses_supported_api_and_clips_composed_actions(self):
        agent = make_agent()
        agent.set_obs_normalizer(make_normalizer())
        env = ActionTrackingEnv()
        evaluator = ResidualEvaluator(FixedChunkPolicy(), agent, .5, 2, 2, 2)
        with patch.object(agent, 'predict_action', return_value=np.array([.8], dtype=np.float32)) as predict:
            metrics = evaluator.evaluate(env, n_episodes=1)
        self.assertEqual(metrics['success_rate'], 1.)
        self.assertEqual(predict.call_count, 2)
        self.assertTrue(all(call.kwargs['deterministic'] for call in predict.call_args_list))
        np.testing.assert_array_equal(env.actions, [[1.], [1.]])


if __name__ == '__main__':
    unittest.main()
