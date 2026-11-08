import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hindsight_grpo.checkpoint import load_checkpoint, save_checkpoint, SignalHandler


class CheckpointTests(unittest.TestCase):
    def test_optimizer_rng_progress_and_accumulated_gradients_roundtrip(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        model(torch.ones(1, 3)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        model(torch.ones(1, 3)).square().mean().backward()
        saved_grad = {k: v.grad.clone() for k, v in model.named_parameters()}
        progress = {'epoch': 1, 'cursor': 9, 'update': 96, 'accumulation_count': 1, 'wandb_run_id': 'abc'}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'latest.pt'
            save_checkpoint(path, model, optimizer, progress, {'arm': 'root'})
            expected_random = (random.random(), np.random.rand(), torch.rand(2))
            restored = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-5)
            result = load_checkpoint(path, restored, restored_optimizer)
            actual_random = (random.random(), np.random.rand(), torch.rand(2))
            self.assertEqual(result['progress'], progress)
            self.assertEqual(expected_random[:2], actual_random[:2])
            torch.testing.assert_close(expected_random[2], actual_random[2])
            for name, parameter in restored.named_parameters():
                torch.testing.assert_close(parameter.grad, saved_grad[name])
            optimizer.step()
            restored_optimizer.step()
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)

    def test_failed_atomic_save_preserves_last_good_checkpoint(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'latest.pt'
            save_checkpoint(path, model, optimizer, {'cursor': 4})
            before = path.read_bytes()
            with patch('hindsight_grpo.checkpoint.torch.save', side_effect=RuntimeError('disk failed')):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(path, model, optimizer, {'cursor': 5})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(temporary).glob('.latest.pt.*')), [])

    def test_signal_sets_flag(self):
        handler = SignalHandler()
        handler._handle(15, None)
        self.assertTrue(handler.requested)
        self.assertEqual(handler.signal_number, 15)


if __name__ == '__main__':
    unittest.main()
