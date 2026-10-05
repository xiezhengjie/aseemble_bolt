"""SAC trainer-facing update API regression tests."""

import unittest

import gymnasium as gym
import numpy as np

from training.policy.sac_policy import SACPolicy


class SACPolicyTest(unittest.TestCase):
    def make_agent(self, **kwargs):
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        config = dict(
            state_dim=3,
            hidden_dim=[16, 16],
            action_dim=2,
            action_space=action_space,
            actor_lr=1e-3,
            critic_lr=2e-3,
            alpha_lr=3e-3,
            tau=0.05,
            gamma=0.99,
            device="cpu",
            autotune=True,
            use_sde=False,
        )
        config.update(kwargs)
        config.pop("device")
        return SACPolicy(**config).to("cpu")

    @staticmethod
    def batch(size=8):
        rng = np.random.default_rng(7)
        return {
            "states": rng.normal(size=(size, 3)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(size, 2)).astype(np.float32),
            "rewards": rng.normal(size=size).astype(np.float32),
            "next_states": rng.normal(size=(size, 3)).astype(np.float32),
            "dones": np.zeros(size, dtype=np.float32),
        }

    def test_cpu_update_runs_and_logs_actor_every_step(self):
        agent = self.make_agent()
        first = agent.update(self.batch(), log_info=True)
        second = agent.update(self.batch(), log_info=True)
        self.assertIn("actor_loss", first)
        self.assertIn("actor_loss", second)
        self.assertTrue(np.isfinite(first["actor_loss"]))
        self.assertTrue(np.isfinite(second["actor_loss"]))

    def test_target_update_and_lr_scale_are_explicit_interfaces(self):
        agent = self.make_agent()
        agent.set_lr_scale(0.4)
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 4e-4)
        self.assertAlmostEqual(agent.critic_optimizer.param_groups[0]["lr"], 8e-4)
        self.assertAlmostEqual(agent.log_alpha_optimizer.param_groups[0]["lr"], 12e-4)
        before = [p.detach().clone() for p in agent.target_critic_1.parameters()]
        agent.critic_1.parameters().__next__().data.add_(1.0)
        agent.update_targets()
        after = list(agent.target_critic_1.parameters())
        self.assertTrue(any(not np.array_equal(a.numpy(), b.detach().numpy()) for a, b in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
