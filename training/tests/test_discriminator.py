"""判别器 BCE 与预测熵正则的回归测试。"""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from training.policy.discriminator_policy import Discriminator


class IndexedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, states, actions):
        return self.logits[states[:, 0].long()].unsqueeze(-1)


class DiscriminatorEntropyTest(unittest.TestCase):
    def _make_discriminator(self, **kwargs):
        return Discriminator(
            state_dim=2,
            action_dim=1,
            hidden_dim=[8],
            lr=1e-3,
            smoothing=0.05,
            **kwargs,
        )

    def test_default_scale_and_loss_decomposition(self):
        discriminator = self._make_discriminator()
        self.assertEqual(discriminator.ent_reg_scale, 0.001)
        discriminator.disc.eval()

        expert_states = np.zeros((4, 2), dtype=np.float32)
        expert_actions = np.zeros((4, 1), dtype=np.float32)
        policy_states = np.ones((4, 2), dtype=np.float32)
        policy_actions = np.ones((4, 1), dtype=np.float32)
        with torch.no_grad():
            expert_logits = discriminator.disc(
                discriminator._as_2d(expert_states),
                discriminator._as_2d(expert_actions),
            )
            policy_logits = discriminator.disc(
                discriminator._as_2d(policy_states),
                discriminator._as_2d(policy_actions),
            )
            bce = (
                F.binary_cross_entropy_with_logits(
                    expert_logits, torch.full_like(expert_logits, 0.05)
                )
                + F.binary_cross_entropy_with_logits(
                    policy_logits, torch.full_like(policy_logits, 0.95)
                )
            )
            logits = torch.cat((expert_logits, policy_logits), dim=0)
            probabilities = torch.sigmoid(logits)
            entropy = -(
                probabilities * probabilities.log()
                + (1.0 - probabilities) * (1.0 - probabilities).log()
            ).mean()

        info = discriminator.update(
            expert_states, expert_actions, policy_states, policy_actions
        )
        self.assertAlmostEqual(info["bce_loss"], bce.item(), places=6)
        self.assertAlmostEqual(info["entropy"], entropy.item(), places=6)
        self.assertAlmostEqual(info["entropy_loss"], -0.001 * entropy.item(), places=6)
        self.assertAlmostEqual(info["loss"], bce.item() - 0.001 * entropy.item(), places=6)

    def test_zero_scale_reproduces_bce(self):
        discriminator = self._make_discriminator(ent_reg_scale=0.0)
        states = np.zeros((4, 2), dtype=np.float32)
        actions = np.zeros((4, 1), dtype=np.float32)
        info = discriminator.update(states, actions, states, actions)
        self.assertEqual(info["entropy_loss"], 0.0)
        self.assertAlmostEqual(info["loss"], info["bce_loss"], places=7)

    def test_entropy_gradient_reduces_confidence(self):
        states = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        actions = np.zeros((3, 1), dtype=np.float32)
        gradients = []
        for scale in (0.0, 0.001):
            discriminator = self._make_discriminator(ent_reg_scale=scale)
            discriminator.disc = IndexedLogits([-2.0, 0.0, 2.0])
            discriminator.disc_optim = torch.optim.SGD(discriminator.disc.parameters(), lr=0.0)
            discriminator.update(states, actions, states, actions)
            gradients.append(discriminator.disc.logits.grad.clone())

        entropy_gradient = gradients[1] - gradients[0]
        self.assertLess(entropy_gradient[0].item(), 0.0)
        self.assertEqual(entropy_gradient[1].item(), 0.0)
        self.assertGreater(entropy_gradient[2].item(), 0.0)
        logits = torch.tensor([-2.0, 0.0, 2.0])
        probabilities = torch.sigmoid(logits)
        expected = 0.001 * logits * probabilities * (1.0 - probabilities) / 3
        torch.testing.assert_close(entropy_gradient, expected, atol=1e-7, rtol=1e-3)

    def test_extreme_logits_are_finite(self):
        discriminator = self._make_discriminator()
        discriminator.disc = IndexedLogits([-1000.0, 0.0, 1000.0])
        discriminator.disc_optim = torch.optim.SGD(discriminator.disc.parameters(), lr=0.0)
        states = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        actions = np.zeros((3, 1), dtype=np.float32)
        info = discriminator.update(states, actions, states, actions)
        for value in info.values():
            self.assertTrue(np.isfinite(value), value)
        self.assertTrue(torch.isfinite(discriminator.disc.logits.grad).all())
        self.assertAlmostEqual(info["entropy"], np.log(2.0) / 3, places=6)

    def test_invalid_scale_is_rejected(self):
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._make_discriminator(ent_reg_scale=value)


if __name__ == "__main__":
    unittest.main()
