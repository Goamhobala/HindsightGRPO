import itertools
import math
import unittest

import torch
import torch.nn.functional as F

from hindsight_grpo.objectives import (
    bernoulli_kl_from_logprobs, cf_loss, cf_target, clipped_root_loss,
    log1mexp, root_advantages,
)


class ObjectiveTests(unittest.TestCase):
    def test_leave_one_out_signs_no_variance_scaling(self):
        rewards = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])
        advantages = root_advantages(rewards)
        self.assertEqual(advantages[0, 0].item(), 1.)
        torch.testing.assert_close(advantages[0, 1:], torch.full((7,), -1 / 7))
        torch.testing.assert_close(advantages.sum(-1), torch.zeros(1), atol=1e-6, rtol=0)
        torch.testing.assert_close(root_advantages(torch.ones(2, 8)), torch.zeros(2, 8))

    def test_identical_scores_ratio_and_clipping(self):
        old = torch.log(torch.tensor([[.1, .5], [.2, .4]]))
        current = old.clone().requires_grad_()
        loss = clipped_root_loss(current, old, torch.tensor([1., -1.]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(current.grad).all()))
        self.assertGreater(current.grad.abs().sum().item(), 0)
        changed = old + math.log(2.)
        clipped = clipped_root_loss(changed, old, torch.ones(2))
        self.assertAlmostEqual(clipped.item(), -1.2, places=6)

    def test_masked_token_means_weight_trajectories_equally(self):
        old = torch.zeros(2, 3)
        new = torch.log(torch.tensor([[1., 1., 1.], [.9, .9, .9]]))
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
        self.assertAlmostEqual(clipped_root_loss(new, old, torch.ones(2), mask=mask).item(), -.95, places=6)

    def test_target_directions_and_detachment(self):
        old = torch.tensor(math.log(.7), requires_grad=True)
        target_keep = cf_target(old, -1.)
        target_change = cf_target(old, 1.)
        self.assertGreater(target_keep.exp().item(), .7)
        self.assertLess(target_change.exp().item(), .7)
        torch.testing.assert_close(cf_target(old, 0.).exp(), old.detach().double().exp())
        self.assertFalse(target_keep.requires_grad)

    def test_kl_derivative_matches_probability_difference(self):
        for old_p, delta, current_z in itertools.product((.01, .4, .99), (-1., 0., 1.), (-8., 0., 8.)):
            z = torch.tensor(current_z, dtype=torch.float64, requires_grad=True)
            old = torch.tensor(math.log(old_p), dtype=torch.float64, requires_grad=True)
            loss = cf_loss(F.logsigmoid(z), old, delta)
            grad, = torch.autograd.grad(loss, z)
            target = cf_target(old, delta).exp()
            torch.testing.assert_close(grad, z.sigmoid() - target, rtol=1e-8, atol=1e-10)
            self.assertIsNone(old.grad)
            self.assertTrue(math.isfinite(loss.item()))

    def test_kl_numerical_boundaries(self):
        for target in (float('-inf'), math.log(1e-100), math.log(1 - 1e-15), 0.):
            for current in (math.log(1e-100), math.log(.5), math.log(1 - 1e-15)):
                loss = bernoulli_kl_from_logprobs(torch.tensor(target, dtype=torch.float64),
                                                torch.tensor(current, dtype=torch.float64, requires_grad=True))
                self.assertTrue(math.isfinite(loss.item()))
                self.assertGreaterEqual(loss.item(), -1e-14)
        torch.testing.assert_close(log1mexp(torch.tensor([float('-inf'), 0.])),
                                   torch.tensor([0., float('-inf')]))

    def test_enumerable_confidence_sampler_likelihood_counterexample(self):
        # Independently propose one token at each position; commit the proposal
        # having greatest *untempered* confidence. P(selected j=1,token=0)=.06,
        # whereas its committed-token probability alone is .6.
        probabilities = ((.9, .1), (.6, .4))
        transitions = {}
        for tokens in itertools.product(range(2), repeat=2):
            proposal_probability = math.prod(probabilities[j][tokens[j]] for j in range(2))
            selected = max(range(2), key=lambda j: probabilities[j][tokens[j]])
            event = (selected, tokens[selected])
            transitions[event] = transitions.get(event, 0.) + proposal_probability
        self.assertAlmostEqual(sum(transitions.values()), 1.)
        self.assertAlmostEqual(transitions[(1, 0)], .06)
        self.assertNotAlmostEqual(transitions[(1, 0)], probabilities[1][0])


if __name__ == '__main__':
    unittest.main()
