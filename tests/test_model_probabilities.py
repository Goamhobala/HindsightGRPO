from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from hindsight_grpo.model import selected_token_logp
from hindsight_grpo.objectives import cf_loss, cf_target


class FixedLogitModel(torch.nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0., margin, 0., 0.]))
    def forward(self, ids, **kwargs):
        return SimpleNamespace(logits=self.logits[None, None].expand(1, ids.shape[1], -1))


class ModelProbabilityTests(unittest.TestCase):
    def test_stable_keep_probability_matches_normal_softmax(self):
        trajectory = SimpleNamespace(commitments=[SimpleNamespace(position=1, token=1)],
                                     state_before=lambda index: [2, 0, 0])
        model = FixedLogitModel(1.)
        log_p = selected_token_logp(model, trajectory, 0, .8)
        expected = F.log_softmax(model.logits.double() / .8, -1)[1]
        torch.testing.assert_close(log_p, expected, atol=1e-14, rtol=1e-14)

    def test_confident_tokens_keep_nonzero_change_mass_and_finite_cf_gradient(self):
        trajectory = SimpleNamespace(commitments=[SimpleNamespace(position=1, token=1)],
                                     state_before=lambda index: [2, 0, 0])
        for margin in (20., 40., 80.):
            with self.subTest(margin=margin):
                model = FixedLogitModel(margin)
                # Demonstrate the original float32 operation loses the complement.
                self.assertEqual(F.log_softmax(model.logits / .8, -1)[1].item(), 0.)
                current = selected_token_logp(model, trajectory, 0, .8)
                self.assertLess(current.item(), 0.)
                self.assertTrue(torch.isfinite(current))
                target = cf_target(current.detach(), 1.)
                self.assertLess(target.item(), current.item())
                loss = cf_loss(current, current.detach(), 1.)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(loss.item(), 0.)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(model.logits.grad).all()))
                self.assertGreater(model.logits.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
