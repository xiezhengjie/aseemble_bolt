"""策略多次更新与判别器交替顺序的回归测试。"""

import unittest
from unittest.mock import Mock, call

import numpy as np
from gymnasium.vector import AutoresetMode

from training.common.train_utils import OffPolicyTrainer, GAILTrainer, ResidualGAILTrainer


class PolicyUpdatesTest(unittest.TestCase):
    def make_trainer(self, trainer_class=OffPolicyTrainer, **kwargs):
        env = Mock()
        env.autoreset_mode = AutoresetMode.SAME_STEP
        env.num_envs = 2
        env.single_observation_space.shape = (2,)
        env.single_action_space.shape = (1,)
        agent = Mock()
        replay = Mock()
        batch_index = 0

        def sample(batch_size):
            nonlocal batch_index
            batch_index += 1
            obs_dim = 3 if trainer_class is ResidualGAILTrainer else 2
            return dict(
                obs=np.full((batch_size, obs_dim), batch_index, dtype=np.float32),
                next_obs=np.zeros((batch_size, 2), dtype=np.float32),
                actions=np.zeros((batch_size, 1), dtype=np.float32),
                res_actions=np.zeros((batch_size, 1), dtype=np.float32),
                rewards=np.zeros(batch_size), successes=np.zeros(batch_size),
            )

        replay.sample.side_effect = sample
        agent.update.side_effect = lambda batch, log_info: (
            {"critic_loss": float(batch["states"][0, 0])} if log_info else {}
        )
        config = dict(
            total_timesteps=8, batch_size=4, is_draw=False, is_save_model=False,
        )
        if trainer_class is not OffPolicyTrainer:
            discriminator = Mock()
            discriminator.predict_rewards.return_value = np.ones(4)
            discriminator.update.side_effect = lambda *args, log_info: (
                {"loss": 0.5} if log_info else {}
            )
            expert = Mock()
            expert.sample.return_value = dict(obs=np.zeros((4, 2)), actions=np.zeros((4, 1)))
            generator = Mock()
            generator.sample.return_value = (np.zeros((4, 2)), np.zeros((4, 1)))
            config.update(
                discriminator=discriminator, expert_buffer=expert,
                generator_buffer=generator, disc_updates=2, disc_batch_size=4,
                gail_reward_scale=False,
            )
        config.update(kwargs)
        if trainer_class is ResidualGAILTrainer:
            trainer = trainer_class(
                env, object(), Mock(), agent, replay,
                residual_scale=0.5, obs_horizon=2, action_interval=2, sampling_steps=2,
                **config,
            )
        else:
            trainer = trainer_class(env, object(), agent, replay, **config)
        trainer.global_step = 6
        return trainer

    def test_default_updates_once(self):
        trainer = self.make_trainer()
        info = trainer._update(log_info=True)
        self.assertEqual(trainer.policy_updates, 1)
        self.assertEqual(trainer.agent.update.call_count, 1)
        self.assertEqual(info, {"critic_loss": 1.0})
        trainer.agent.set_lr_scale.assert_called_once_with(0.3)
        self.assertEqual(trainer.agent.update_targets.call_count, 1)

    def test_multiple_updates_resample_and_log_last(self):
        trainer = self.make_trainer(policy_updates=3)
        info = trainer._update(log_info=True)
        self.assertEqual(trainer.replay_buffer.sample.call_args_list, [call(4)] * 3)
        calls = trainer.agent.update.call_args_list
        self.assertEqual([c.kwargs["log_info"] for c in calls], [False, False, True])
        self.assertEqual([c.args[0]["states"][0, 0] for c in calls], [1, 2, 3])
        self.assertEqual(info, {"critic_loss": 3.0})
        trainer.agent.set_lr_scale.assert_called_once_with(0.3)
        self.assertEqual(trainer.agent.update_targets.call_count, 3)

    def test_policy_updates_finish_before_discriminator(self):
        for trainer_class in (GAILTrainer, ResidualGAILTrainer):
            with self.subTest(trainer_class=trainer_class):
                trainer = self.make_trainer(trainer_class, policy_updates=3)
                events = []
                policy_update = trainer.agent.update.side_effect
                disc_update = trainer.discriminator.update.side_effect

                def record_policy(batch, log_info):
                    events.append("policy")
                    if trainer_class is ResidualGAILTrainer:
                        self.assertEqual(batch["states"].shape, (4, 2))
                    return policy_update(batch, log_info)

                def record_discriminator(*args, log_info):
                    events.append("discriminator")
                    return disc_update(*args, log_info=log_info)

                trainer.agent.update.side_effect = record_policy
                trainer.discriminator.update.side_effect = record_discriminator
                info = trainer._update(log_info=True)
                self.assertEqual(events, ["policy"] * 3 + ["discriminator"] * 2)
                self.assertEqual(trainer.discriminator.predict_rewards.call_count, 3)
                self.assertEqual(trainer.expert_buffer.sample.call_count, 2)
                self.assertEqual(trainer.generator_buffer.sample.call_count, 2)
                self.assertEqual(info, {"critic_loss": 3.0, "disc_loss": 0.5})
                self.assertEqual(trainer.disc_losses, [0.5])
                trainer.agent.set_lr_scale.assert_called_once_with(0.3)
                self.assertEqual(trainer.agent.update_targets.call_count, 3)
                trainer.discriminator.set_lr_scale.assert_called_once_with(0.25)

    def test_target_updates_follow_cumulative_gradient_steps(self):
        trainer = self.make_trainer(policy_updates=3, target_update_interval=2)
        trainer._update(log_info=False)
        self.assertEqual(trainer.gradient_step, 3)
        self.assertEqual(trainer.agent.update_targets.call_count, 1)
        trainer._update(log_info=False)
        self.assertEqual(trainer.gradient_step, 6)
        self.assertEqual(trainer.agent.update_targets.call_count, 3)

    def test_custom_policy_lr_floor_is_applied(self):
        trainer = self.make_trainer(policy_lr_min_ratio=0.8)
        trainer._update(log_info=False)
        trainer.agent.set_lr_scale.assert_called_once_with(0.8)

    def test_invalid_target_interval_and_lr_floor(self):
        for kwargs, message in (
            ({"target_update_interval": 0}, "target_update_interval"),
            ({"target_update_interval": True}, "target_update_interval"),
            ({"policy_lr_min_ratio": -0.1}, "policy_lr_min_ratio"),
            ({"policy_lr_min_ratio": float("nan")}, "policy_lr_min_ratio"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.make_trainer(**kwargs)

    def test_logging_disabled_for_all_updates(self):
        trainer = self.make_trainer(GAILTrainer, policy_updates=3)
        self.assertEqual(trainer._update(log_info=False), {})
        for c in trainer.agent.update.call_args_list + trainer.discriminator.update.call_args_list:
            self.assertFalse(c.kwargs["log_info"])
        self.assertEqual(trainer.disc_losses, [])

    def test_invalid_update_counts(self):
        for value in (0, -1, 1.5, 2.0, True, "2", None, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "policy_updates"):
                self.make_trainer(policy_updates=value)


if __name__ == "__main__":
    unittest.main()
