"""序列边界、DataLoader 训练及跨 action chunk 的先验回归测试。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from training.common.buffer_utils import (
    ExpertBuffer, ExpertDataManager, ExpertSequenceDataset, SequenceSampler,
    create_indices,
)
from training.model.base.conditional_unet1d import ConditionalUnet1D
from training.policy.diffusion_policy import DiffusionPolicy


def make_policy():
    return DiffusionPolicy(
        model=ConditionalUnet1D(
            input_dim=1, global_cond_dim=2, diffusion_step_embed_dim=16,
            down_dims=[16, 32], n_groups=8,
        ),
        noise_scheduler=DDPMScheduler(num_train_timesteps=100),
        horizon=8, obs_dim=1, action_dim=1, n_action_steps=4,
        n_obs_steps=2, num_inference_steps=4,
    )


class SequenceDataTests(unittest.TestCase):
    def test_padding_never_crosses_episode_boundary(self):
        buffer = ExpertBuffer(20)
        values = np.array([0, 1, 2, 10, 11], dtype=np.float32)[:, None]
        buffer.add_batch(values, values + 100, [0, 0, 1, 0, 0])
        sampler = SequenceSampler(buffer, 4, pad_before=1, pad_after=2)
        self.assertEqual(len(sampler), 5)
        expected = [[0, 0, 1, 2], [0, 1, 2, 2], [1, 2, 2, 2],
                    [10, 10, 11, 11], [10, 11, 11, 11]]
        for index, sequence in enumerate(expected):
            np.testing.assert_array_equal(sampler.sample_sequence(index)['obs'][:, 0], sequence)
        dataset = ExpertSequenceDataset(buffer, 4, 2, 1, 2)
        for sample in dataset:
            self.assertTrue(torch.isfinite(sample['obs']).all())
        np.testing.assert_array_equal(dataset[0]['obs'][:, 0], [0, 0])
        np.testing.assert_array_equal(dataset[0]['action'][:, 0], [100, 101, 102, 102])

    def test_ring_wrap_uses_chronological_data(self):
        buffer = ExpertBuffer(4)
        values = np.arange(6, dtype=np.float32)[:, None]
        buffer.add_batch(values, values, [0, 0, 1, 0, 0, 1])
        sampler = SequenceSampler(buffer, 2, pad_before=1)
        np.testing.assert_array_equal(sampler.sample_sequence(0)['obs'][:, 0], [2, 2])
        np.testing.assert_array_equal(sampler.sample_sequence(1)['obs'][:, 0], [3, 3])
        np.testing.assert_array_equal(sampler.sample_sequence(3)['obs'][:, 0], [4, 5])

    def test_empty_and_invalid_masks(self):
        ends = np.array([1, 2])
        indices = create_indices(ends, 8, np.ones(2, dtype=bool))
        self.assertEqual(indices.shape, (0, 4))
        indices = create_indices(ends, 1, np.zeros(2, dtype=bool))
        self.assertEqual(indices.shape, (0, 4))
        with self.assertRaises(ValueError):
            create_indices(ends, 1, np.ones(3, dtype=bool))

    def test_dataloader_split_and_real_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'expert.npz'
            values = np.repeat(np.arange(4, dtype=np.float32), 10)[:, None]
            np.savez(path, states=values, actions=values / 4,
                     dones=np.tile([0] * 9 + [1], 4))
            manager = ExpertDataManager(capacity=3)
            manager.load_data(path)
            self.assertEqual(manager.buffer.size(), 40)
            loaders = manager.trans_dataloader(
                .5, 3, seed=42, horizon=8, n_obs_steps=2, n_action_steps=4,
            )
            self.assertTrue(all(isinstance(loader, DataLoader) for loader in loaders))
            episode_ids = []
            for loader in loaders:
                ids = set()
                for batch in loader:
                    self.assertEqual(batch['obs'].shape[1:], (2, 1))
                    self.assertEqual(batch['action'].shape[1:], (8, 1))
                    self.assertTrue(torch.isfinite(batch['obs']).all())
                    ids.update(batch['obs'][:, 0, 0].tolist())
                episode_ids.append(ids)
            self.assertFalse(episode_ids[0] & episode_ids[1])
            self.assertEqual(episode_ids[0] | episode_ids[1], {0, 1, 2, 3})
            repeat = manager.trans_dataloader(
                .5, 3, seed=42, horizon=8, n_obs_steps=2, n_action_steps=4,
            )
            np.testing.assert_array_equal(loaders[0].dataset.sampler.indices,
                                          repeat[0].dataset.sampler.indices)
            policy = make_policy()
            before = next(policy.model.parameters()).detach().clone()
            self.assertTrue(np.isfinite(policy.fit_epoch(loaders[0])))
            self.assertFalse(torch.equal(before, next(policy.model.parameters())))
            self.assertTrue(np.isfinite(policy.eval_epoch(loaders[1])))


class MotionPriorTests(unittest.TestCase):
    def test_previous_prediction_is_shifted_and_noised(self):
        policy = make_policy()
        obs = torch.zeros(2, 2, 1)
        scheduler = policy.inference_noise_scheduler
        captured = []
        add_noise = scheduler.add_noise

        def record(prior, noise, timestep):
            captured.append((prior.clone(), noise.clone(), timestep.clone()))
            return add_noise(prior, noise, timestep)

        with patch.object(scheduler, 'add_noise', side_effect=record):
            first = policy.predict_action({'obs': obs})
            second = policy.predict_action({'obs': obs})
        self.assertEqual(first['action'].shape, (2, 4, 1))
        torch.testing.assert_close(first['action'], first['action_full'][:, :4])
        torch.testing.assert_close(captured[0][0], torch.zeros(2, 8, 1))
        torch.testing.assert_close(captured[1][0][:, :4], first['action_full'][:, 4:])
        torch.testing.assert_close(captured[1][0][:, 4:], torch.zeros(2, 4, 1))
        self.assertTrue((captured[1][2] == 50).all())
        self.assertTrue(torch.isfinite(second['action']).all())
        self.assertNotIn('prev_naction', policy.state_dict())

        policy.reset()
        self.assertIsNone(policy.prev_naction)
        with patch.object(scheduler, 'add_noise', side_effect=record):
            policy.sample(obs)
            policy.sample(obs[:1])
        torch.testing.assert_close(captured[2][0], torch.zeros(2, 8, 1))
        torch.testing.assert_close(captured[3][0], torch.zeros(1, 8, 1))

    def test_matches_reference_ddim_loop(self):
        policy = make_policy()
        condition = torch.zeros(1, 8, 1)
        global_cond = torch.zeros(1, 2)
        scheduler = policy.inference_noise_scheduler
        for _ in range(2):
            prior = torch.zeros_like(condition) if policy.prev_naction is None else policy.prev_naction.clone()
            noise = torch.randn(condition.shape, generator=torch.Generator().manual_seed(7))
            scheduler.set_timesteps(4)
            expected = scheduler.add_noise(prior, noise, torch.tensor([50]))
            with torch.no_grad():
                for t in scheduler.timesteps:
                    pred = policy.model(expected, t, global_cond=global_cond)
                    expected = scheduler.step(pred, t, expected, eta=0.0).prev_sample
            actual = policy.conditional_sample(
                condition, torch.zeros_like(condition, dtype=torch.bool),
                global_cond=global_cond, generator=torch.Generator().manual_seed(7),
            )
            torch.testing.assert_close(actual, expected)

    def test_condition_mask_is_preserved(self):
        policy = make_policy()
        data = torch.zeros(1, 8, 1)
        data[:, :2] = .25
        mask = torch.zeros_like(data, dtype=torch.bool)
        mask[:, :2] = True
        result = policy.conditional_sample(data, mask, global_cond=torch.zeros(1, 2))
        torch.testing.assert_close(result[mask], data[mask])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
