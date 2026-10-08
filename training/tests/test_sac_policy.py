"""SAC trainer-facing update API regression tests."""

import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from training.policy.sac_policy import SACPolicy, ResidualSACPolicy


class SACPolicyTest(unittest.TestCase):
    def make_agent(self, policy_class=SACPolicy, **kwargs):
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        config = dict(
            state_dim=3,
            hidden_dim=[16, 16],
            action_dim=2,
            action_space=action_space,
            optimizer={
                name: dict(lr=lr, betas=[0.9, 0.999], eps=1e-5, weight_decay=0.0)
                for name, lr in (("actor", 1e-3), ("critic", 2e-3), ("alpha", 3e-3))
            },
            lr_scheduler=dict(name="constant", num_warmup_steps=0),
            num_training_steps=10,
            tau=0.05,
            gamma=0.99,
            autotune=True,
            use_sde=False,
        )
        config.update(kwargs)
        return policy_class(**config).to("cpu")

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

    def test_target_update_is_an_explicit_interface(self):
        agent = self.make_agent()
        before = [p.detach().clone() for p in agent.target_critic_1.parameters()]
        agent.critic_1.parameters().__next__().data.add_(1.0)
        agent.update_targets()
        after = list(agent.target_critic_1.parameters())
        self.assertTrue(any(not np.array_equal(a.numpy(), b.detach().numpy()) for a, b in zip(before, after)))

    def test_default_grad_clip_norms_preserve_existing_thresholds(self):
        agent = self.make_agent()
        self.assertEqual(agent.actor_grad_clip_norm, 1.0)
        self.assertEqual(agent.critic_grad_clip_norm, 5.0)

    def test_custom_grad_clip_norms_are_applied_to_both_policy_types(self):
        clip_grad_norm = torch.nn.utils.clip_grad_norm_
        for policy_class in (SACPolicy, ResidualSACPolicy):
            with self.subTest(policy=policy_class.__name__):
                agent = self.make_agent(
                    policy_class, actor_grad_clip_norm=0.01, critic_grad_clip_norm=0.02,
                )
                clipped_norms = []
                parameter_ids = []

                def clip(parameters, max_norm):
                    parameters = list(parameters)
                    parameter_ids.append({id(parameter) for parameter in parameters})
                    original_norm = clip_grad_norm(parameters, max_norm)
                    clipped_norms.append(torch.linalg.vector_norm(torch.stack([
                        parameter.grad.norm() for parameter in parameters
                        if parameter.grad is not None
                    ])).item())
                    return original_norm

                with patch("training.policy.sac_policy.torch.nn.utils.clip_grad_norm_",
                           side_effect=clip) as clipping:
                    info = agent.update(self.batch(), log_info=True)

                self.assertEqual([call.args[1] for call in clipping.call_args_list], [0.02, 0.01])
                self.assertEqual(parameter_ids[0], {
                    id(parameter) for critic in (agent.critic_1, agent.critic_2)
                    for parameter in critic.parameters()
                })
                self.assertEqual(parameter_ids[1], {id(parameter) for parameter in agent.actor.parameters()})
                for logged, clipped, limit in zip(
                    (info["gn_critic"], info["gn_actor"]), clipped_norms, (0.02, 0.01),
                ):
                    self.assertGreater(logged, limit)
                    self.assertLessEqual(clipped, limit + 1e-6)

    def test_grad_clip_norms_must_be_positive_and_finite(self):
        for name in ("actor_grad_clip_norm", "critic_grad_clip_norm"):
            for value in (0.0, -1.0, float("inf"), float("nan")):
                with self.subTest(parameter=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.make_agent(**{name: value})

    def test_yaml_grad_clip_norms_are_passed_through_hydra(self):
        config_dir = Path(__file__).resolve().parents[2] / "config"
        for name in ("train_sac", "train_sac_gail", "train_sac_gail_residual"):
            with self.subTest(config=name):
                config = OmegaConf.load(config_dir / f"{name}.yaml").policy
                self.assertEqual(config.actor_grad_clip_norm, 1.0)
                self.assertEqual(config.critic_grad_clip_norm, 5.0)
                config.actor_grad_clip_norm = 2.0
                config.critic_grad_clip_norm = 10.0
                config.hidden_dim = [16, 16]
                agent = instantiate(
                    config, state_dim=3, action_dim=2, num_training_steps=10,
                    action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                )
                expected_class = ResidualSACPolicy if name.endswith("residual") else SACPolicy
                self.assertIsInstance(agent, expected_class)
                self.assertEqual(agent.actor_grad_clip_norm, 2.0)
                self.assertEqual(agent.critic_grad_clip_norm, 10.0)


if __name__ == "__main__":
    unittest.main()
