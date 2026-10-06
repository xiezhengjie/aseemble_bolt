"""无需 MuJoCo 的采集与训练回归测试。"""

import importlib
import random
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from training.common.eval_utils import Evaluator, ResidualEvaluator
from training.common.buffer_utils import ExpertBuffer, GenWindowView, ReplayBuffer, ResidualReplayBuffer
from training.common.rl_utils import EpisodeStatsWrapper
from training.common.train_utils import DataCollector, GAILTrainer, OffPolicyTrainer, ResidualDataCollector, ResidualGAILTrainer
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
        self.assertEqual(evaluator.evaluate.call_count, 2)
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


class EvaluationSeedTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.evaluator = Mock()
        self.stats = dict(success_rate=0.0, return_mean=0.0,
                          return_std=0.0, peak_force_mean=0.0)
        self.evaluator.evaluate.return_value = self.stats
        self.trainer = OffPolicyTrainer(
            Mock(), object(), Mock(), Mock(), 4,
            collector=Mock(), evaluator=self.evaluator,
            learning_starts=10, eval_interval=2,
            eval_episodes=2, final_eval_episodes=3,
            is_save_model=False, is_draw=False,
        )

    @staticmethod
    def draw_random_values():
        return random.random(), float(np.random.random()), float(torch.rand(()))

    def test_periodic_evaluation_continues_global_rng_streams(self):
        samples = []

        def evaluate(*args, **kwargs):
            samples.append(self.draw_random_values())
            return self.stats

        self.evaluator.evaluate.side_effect = evaluate
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        python_rng = random.Random(seed)
        numpy_rng = np.random.RandomState(seed)
        torch_rng = torch.Generator().manual_seed(seed)
        expected = [(python_rng.random(), float(numpy_rng.random()),
                     float(torch.rand((), generator=torch_rng))) for _ in range(3)]
        self.trainer._evaluate()
        self.trainer._evaluate()
        self.assertEqual(samples, expected[:2])
        self.assertEqual(self.draw_random_values(), expected[2])
        self.evaluator.evaluate.assert_has_calls([
            call(self.trainer.eval_env, n_episodes=2, seed_offset=None),
            call(self.trainer.eval_env, n_episodes=2, seed_offset=None),
        ])

    def test_environment_rng_continues_across_periodic_evaluations(self):
        class RandomEnv(CounterEnv):
            def reset(self, *, seed=None, options=None):
                obs, info = super().reset(seed=seed, options=options)
                samples.append(float(self.np_random.random()))
                seeds.append(seed)
                return obs, info

        samples, seeds = [], []
        env = RandomEnv(1)
        self.addCleanup(env.close)
        env.reset(seed=100000)
        evaluator = Evaluator(Mock())
        evaluator.evaluate(env, n_episodes=2, seed_offset=None)
        evaluator.evaluate(env, n_episodes=2, seed_offset=None)
        self.assertEqual(seeds, [100000, None, None, None, None])
        np.testing.assert_array_equal(samples, np.random.default_rng(100000).random(5))
        evaluator.evaluate(env, n_episodes=2, seed_offset=200000)
        self.assertEqual(seeds[-2:], [200000, 200001])

    def test_training_only_runs_periodic_evaluations(self):
        self.trainer.env.num_envs = 1
        self.trainer.collector.step.return_value = []
        self.trainer.learning_starts = 0
        self.trainer._update = Mock(return_value={})
        self.trainer._save_checkpoint = Mock()
        self.trainer.wb_run = Mock()
        self.trainer.train()
        self.assertEqual(self.evaluator.evaluate.call_count, 2)
        self.trainer._save_checkpoint.assert_any_call("final_model", {})
        self.assertEqual(self.trainer.final_stats, {})
        self.trainer.wb_run.finish.assert_not_called()

    def test_final_results_are_logged_and_saved(self):
        self.trainer.wb_run = Mock()
        self.trainer._save_checkpoint = Mock()
        logged = []
        self.trainer.wb_run.log.side_effect = lambda values, step: logged.append((dict(values), step))
        self.trainer.record_final_evaluation(self.stats)
        self.assertIs(self.trainer.final_stats, self.stats)
        self.trainer._save_checkpoint.assert_called_once_with("final_model", self.stats)
        self.assertEqual(logged, [({
            "final_eval/success_rate": 0.0,
            "final_eval/return_mean": 0.0,
            "final_eval/peak_force_mean": 0.0,
        }, 0)])


class TrainingEntrypointEvaluationTest(unittest.TestCase):
    def test_all_configs_use_the_same_final_evaluation_conditions(self):
        from omegaconf import OmegaConf

        root = Path(__file__).resolve().parents[2]
        conditions = []
        for name in ("train_base_policy", "train_sac", "train_sac_gail", "train_sac_gail_residual"):
            cfg = OmegaConf.load(root / "config" / f"{name}.yaml")
            episodes = (cfg.training.eval_episodes if name == "train_base_policy"
                        else cfg.trainer.final_eval_episodes)
            conditions.append((int(cfg.training.seed), int(cfg.training.eval_seed),
                               int(cfg.training.eval_env_seed), int(episodes)))
        self.assertTrue(all(condition == conditions[0] for condition in conditions))
        self.assertNotEqual(conditions[0][1], conditions[0][2])

    def test_base_entrypoint_separates_global_and_environment_evaluation_seeds(self):
        from omegaconf import OmegaConf

        module = importlib.import_module("training.train.train_base_policy")
        root = Path(__file__).resolve().parents[2]
        cfg = OmegaConf.load(root / "config" / "train_base_policy.yaml")
        cfg.logging.name = "test"
        manager, train_loader, val_loader, policy, trainer, env, evaluator, run = (
            Mock() for _ in range(8))
        manager.trans_dataloader.return_value = (train_loader, val_loader)
        train_loader.dataset.sampler.replay_buffer = {'obs': np.zeros((2, 10))}
        train_loader.dataset.sampler.indices = [(0, 2, 0, 2)]
        policy.to.return_value = policy
        with patch.object(module, "ExpertDataManager", return_value=manager), \
                patch.object(module.hydra.utils, "instantiate", return_value=policy), \
                patch.object(module, "SupervisedPolicyTrainer", return_value=trainer), \
                patch.object(module, "AssembleMuJoCoEnv", return_value=env), \
                patch.object(module, "EpisodeStatsWrapper", side_effect=lambda value: value), \
                patch.object(module, "BaseChunkPolicyEvaluator", return_value=evaluator), \
                patch.object(module.wandb, "init", return_value=run), \
                patch.object(module.torch.cuda, "is_available", return_value=False), \
                patch.object(module, "set_seed") as seed:
            module.main.__wrapped__(cfg)
        seed.assert_has_calls([call(int(cfg.training.seed)), call(int(cfg.training.eval_seed))])
        self.assertEqual(seed.call_count, 2)
        evaluator.evaluate.assert_called_once_with(
            env, n_episodes=int(cfg.training.eval_episodes),
            seed_offset=int(cfg.training.eval_env_seed))
        trainer.train.assert_called_once_with()
        env.close.assert_called_once_with()
        run.finish.assert_called_once_with()

    def test_final_evaluation_and_cleanup_in_each_entrypoint(self):
        from omegaconf import OmegaConf

        root = Path(__file__).resolve().parents[2]
        for name in ("train_sac", "train_sac_gail", "train_sac_gail_residual"):
            module = importlib.import_module(f"training.train.{name}")
            cfg = OmegaConf.load(root / "config" / f"{name}.yaml")
            cfg.logging.name = "test"
            for save_model, fails in ((True, False), (False, False), (True, True)):
                with self.subTest(entrypoint=name, save_model=save_model, fails=fails):
                    env, eval_env, agent, trainer, run = (Mock() for _ in range(5))
                    base_policy, disc = Mock(), Mock()
                    base_policy.to.return_value = base_policy
                    disc.to.return_value = disc
                    env.single_observation_space.shape = (10,)
                    env.single_action_space.shape = (6,)
                    env.single_action_space.dtype = np.dtype("float32")
                    trainer.is_save_model = save_model
                    trainer.final_eval_episodes = 3
                    metrics = dict(success_rate=1.0)
                    trainer.evaluator.evaluate.return_value = metrics
                    if fails:
                        trainer.evaluator.evaluate.side_effect = RuntimeError("evaluation failed")
                    events = Mock()
                    events.attach_mock(trainer.train, "train")
                    events.attach_mock(agent.load_model, "load")
                    events.attach_mock(trainer.evaluator.evaluate, "evaluate")

                    def instantiate(config, **kwargs):
                        if config is cfg.trainer:
                            return trainer
                        if config is cfg.policy:
                            agent.to.return_value = agent
                            return agent
                        if name == "train_sac_gail_residual" and config is cfg.base_policy:
                            return base_policy
                        if name != "train_sac" and config is cfg.discriminator:
                            return disc
                        return Mock()

                    with patch.object(module.gym.vector, "AsyncVectorEnv", return_value=env), \
                            patch.object(module, "make_env", return_value=lambda: eval_env), \
                            patch.object(module.hydra.utils, "instantiate", side_effect=instantiate), \
                            patch.object(module.wandb, "init", return_value=run), \
                            patch.object(module.torch.cuda, "is_available", return_value=False), \
                            patch.object(module, "set_seed") as seed:
                        events.attach_mock(seed, "seed")
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                                module.main.__wrapped__(cfg)
                        else:
                            module.main.__wrapped__(cfg)
                            trainer.record_final_evaluation.assert_called_once_with(metrics)
                        seed.assert_has_calls([
                            call(int(cfg.training.seed)), call(int(cfg.training.eval_seed)),
                        ])
                        self.assertEqual(seed.call_count, 2)
                        eval_env.reset.assert_called_once_with(seed=int(cfg.training.eval_env_seed))
                        expected = [call.train(), call.seed(int(cfg.training.eval_seed))]
                        if save_model:
                            expected.append(call.load(module.MODE_DIR / "final_model"))
                        else:
                            agent.load_model.assert_not_called()
                        expected.append(call.evaluate(eval_env, n_episodes=3,
                                                      seed_offset=int(cfg.training.eval_env_seed)))
                        events.assert_has_calls(expected)
                    env.close.assert_called_once_with()
                    eval_env.close.assert_called_once_with()
                    run.finish.assert_called_once_with()
                    if name == "train_sac_gail_residual":
                        base_policy.load_model.assert_called_once_with(module.BASE_POLICY_DIR)
                        base_policy.requires_grad_.assert_called_once_with(False)
                        base_policy.eval.assert_called_once_with()
                        agent.set_obs_normalizer.assert_called_once_with(base_policy.obs_normalizer)
                        disc.set_obs_normalizer.assert_called_once_with(base_policy.obs_normalizer)

    def test_residual_entrypoint_rejects_missing_stats_before_creating_resources(self):
        from omegaconf import OmegaConf

        module = importlib.import_module("training.train.train_sac_gail_residual")
        root = Path(__file__).resolve().parents[2]
        cfg = OmegaConf.load(root / "config" / "train_sac_gail_residual.yaml")
        base_policy = Mock()
        base_policy.to.return_value = base_policy
        base_policy.obs_normalizer = None
        with patch.object(module.hydra.utils, "instantiate", return_value=base_policy) as instantiate, \
                patch.object(module.wandb, "init") as init_run, \
                patch.object(module.gym.vector, "AsyncVectorEnv") as vector_env, \
                patch.object(module, "make_env") as make_env, \
                patch.object(module.torch.cuda, "is_available", return_value=False), \
                patch.object(module, "set_seed"):
            with self.assertRaisesRegex(ValueError, "缺少观测归一化统计量"):
                module.main.__wrapped__(cfg)
        instantiate.assert_called_once_with(cfg.base_policy)
        base_policy.load_model.assert_called_once_with(module.BASE_POLICY_DIR)
        init_run.assert_not_called()
        vector_env.assert_not_called()
        make_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
