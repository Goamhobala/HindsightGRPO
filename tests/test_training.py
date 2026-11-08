"""Synthetic full-loop checks: no network, GPU allocation, or generated-code exec."""
from contextlib import ExitStack, contextmanager
import copy
from dataclasses import dataclass, asdict
import json
import errno
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hindsight_grpo.checkpoint import atomic_json, load_checkpoint, save_checkpoint
from hindsight_grpo.evaluation import StopRequested
from hindsight_grpo.model import ModelBundle
from hindsight_grpo.train import Experiment


class TinyAdapterModel(torch.nn.Module):
    """A bidirectional model with frozen base weights and a trainable adapter."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 5)
        self.base = torch.nn.Linear(5, 7)
        self.adapter = torch.nn.Linear(5, 7, bias=False)
        self.embedding.requires_grad_(False)
        self.base.requires_grad_(False)
        self.config = SimpleNamespace(eos_token_id=None)
        self.seen_inputs = []
    def get_input_embeddings(self):
        return self.embedding
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        embeddings = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = embeddings + embeddings.mean(dim=1, keepdim=True)
        logits = self.base(hidden) + self.adapter(hidden)
        # MASK has negligible mass, so every synthetic draw can be committed.
        logits = logits.clone()
        logits[..., 0] = -100.
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    mask_token_id = 0
    def encode(self, text, **kwargs):
        return [1, 2]
    def decode(self, ids, **kwargs):
        return ','.join(str(int(value)) for value in ids)
    def apply_chat_template(self, messages, **kwargs):
        return 'synthetic prompt'


@dataclass
class FakeRewardResult:
    reward: float
    partial: float
    syntax_valid: bool = True
    error: str = ''
    timeout: bool = False
    def to_dict(self):
        return asdict(self)


class FakeReward:
    def __init__(self, **kwargs):
        pass
    def check_available(self):
        return {'synthetic_reward': True}
    def evaluate(self, text, row):
        # Stateless rewards ensure restart doesn't change any reward assignment.
        value = float(sum(map(int, text.split(','))) % 2)
        return FakeRewardResult(value, value)


class FakeTracker:
    def __init__(self, run_dir, arm, config, **kwargs):
        self.run_dir, self.arm = Path(run_dir), arm
        id_path = self.run_dir / 'wandb_run_id'
        self.run_id = id_path.read_text().strip() if id_path.exists() else 'synthetic-run'
        id_path.write_text(self.run_id + '\n')
        atomic_json(self.run_dir / 'config.json', config)
        self.url = 'https://example.invalid/synthetic'
        self.records, self.logs, self.artifacts = [], [], []
    def record(self, kind, value, step=None):
        self.records.append((kind, value, step))
    def log(self, metrics, step):
        self.logs.append((copy.deepcopy(metrics), step))
    def artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))
    def flush_records(self):
        pass
    def summary(self, value):
        self.final_summary = value
    def finish(self, exit_code=0):
        self.finished_exit_code = exit_code


def small_config():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'experiment.json'
    result = json.loads(path.read_text())
    result.update(generation_length=4, denoising_steps=4, block_size=2,
                  evaluation_samples=2, epochs=1)
    return result


def fake_data(**kwargs):
    atomic_json(kwargs['manifest_path'], {'synthetic': True})
    return {'train': [{'task_id': i, 'text': str(i), 'test_list': []} for i in range(4)],
            'validation': [{'task_id': 10, 'text': '', 'test_list': []}],
            'test': [{'task_id': 20, 'text': '', 'test_list': []}]}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict('os.environ', {'SLURM_JOB_ID': 'synthetic'}))
        self.stack.enter_context(patch('hindsight_grpo.train.load_model',
            side_effect=lambda config: ModelBundle(TinyAdapterModel().eval(), TinyTokenizer())))
        self.stack.enter_context(patch('hindsight_grpo.train.Tracker', FakeTracker))
        self.stack.enter_context(patch('hindsight_grpo.train.RewardRunner', FakeReward))
        self.stack.enter_context(patch('hindsight_grpo.train.load_mbpp', fake_data))
        self.stack.enter_context(patch('hindsight_grpo.train.implementation_token_mask',
            side_effect=lambda ids, tokenizer: [True] * len(ids)))
        for name, value in [('is_available', True), ('device_count', 0),
                            ('max_memory_allocated', 0), ('get_rng_state_all', []),
                            ('is_current_stream_capturing', False)]:
            self.stack.enter_context(patch.object(torch.cuda, name, return_value=value))
        for name in ('manual_seed_all', 'synchronize', 'set_rng_state_all'):
            self.stack.enter_context(patch.object(torch.cuda, name))
    def tearDown(self):
        self.stack.close()
        torch.set_num_threads(self.previous_threads)
    def create(self, directory, arm='root', verify=True):
        experiment = Experiment(small_config(), arm, directory, 'synthetic', verify=verify)
        self.directory_resources.enter_context(experiment)
        return experiment
    def close(self, experiment):
        experiment.close()

    @contextmanager
    def temporary_directory(self):
        # Resource callbacks run BEFORE TemporaryDirectory removes files. On
        # NFS an unclosed .training.lock becomes a .nfs file and can replace the
        # useful optimizer/assertion error with a Directory-not-empty error.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as resources:
            previous = getattr(self, 'directory_resources', None)
            self.directory_resources = resources
            try:
                yield directory
            finally:
                self.directory_resources = previous

    def test_one_optimizer_update_per_arm_has_finite_nonzero_adapter_gradient(self):
        for arm in ('root', 'entropy_cf', 'r_cf', 'gradient_ce_cf'):
            with self.subTest(arm=arm), self.temporary_directory() as directory:
                experiment = self.create(directory, arm)
                before = {name: value.clone() for name, value in experiment.model.state_dict().items()}
                self.assertEqual(experiment.run(), 0)
                self.assertEqual(experiment.progress['update'], 1)
                self.assertEqual(experiment.progress['prompts_processed'], 4)
                metrics = experiment.tracker.logs[-1][0]
                self.assertTrue(0 < metrics['train/gradient_norm'] < float('inf'))
                self.assertFalse(torch.equal(before['adapter.weight'], experiment.model.adapter.weight))
                for name in ('embedding.weight', 'base.weight', 'base.bias'):
                    torch.testing.assert_close(before[name], experiment.model.state_dict()[name], rtol=0, atol=0)
                for name, value in experiment.rollout.state_dict().items():
                    torch.testing.assert_close(before[name], value, rtol=0, atol=0)
                self.assertTrue(all(parameter.grad is None for parameter in experiment.rollout.parameters()))
                if arm != 'root':
                    self.assertEqual(metrics['cf/valid_states'], 4)
                    cf = [record for kind, record, step in experiment.tracker.records if kind == 'counterfactual']
                    self.assertEqual(len(cf), 4)
                    self.assertTrue(all(len(item['keep']) == 4 and len(item['alternative']) == 8 for item in cf))
                self.close(experiment)

    def test_cpu_optimizer_never_queries_the_actual_cuda_driver(self):
        with self.temporary_directory() as directory:
            experiment = self.create(directory)
            # Simulate a Linux CUDA-enabled torch wheel on a login node whose
            # driver is absent/old. These synthetic parameters are all CPU-only.
            with patch('torch.backends.cuda.is_built', return_value=True), \
                 patch('torch.cuda.graphs._cuda_isCurrentStreamCapturing', create=True,
                       side_effect=RuntimeError('real CUDA driver must not be touched')) as native_query:
                self.assertEqual(experiment.run(), 0)
                self.assertTrue(torch.cuda.is_current_stream_capturing.called)
                native_query.assert_not_called()

    def test_training_exception_closes_lock_before_nfs_style_directory_cleanup(self):
        experiment = None
        original_rmtree = shutil.rmtree
        def nfs_rmtree(path, *args, **kwargs):
            # NFS retains an unlinked open lock as .nfs*, so deleting its parent
            # fails. Simulate that rule while preserving ordinary cleanup.
            if experiment is not None and not experiment.lock.closed:
                raise OSError(errno.ENOTEMPTY, 'open .nfs lock file', str(path))
            return original_rmtree(path, *args, **kwargs)
        with patch('tempfile._shutil.rmtree', side_effect=nfs_rmtree):
            with self.assertRaisesRegex(RuntimeError, 'primary optimizer failure'):
                with self.temporary_directory() as directory:
                    experiment = self.create(directory)
                    experiment.collect_prompt = Mock(side_effect=RuntimeError('primary optimizer failure'))
                    experiment.run()
        self.assertTrue(experiment.lock.closed)
        self.assertEqual(experiment.tracker.finished_exit_code, 1)

    def test_initialization_failure_closes_lock_and_started_tracker(self):
        handles, trackers = [], []
        original_open = Path.open
        def track_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if path.name == '.training.lock':
                handles.append(handle)
            return handle
        def tracker(*args, **kwargs):
            result = FakeTracker(*args, **kwargs)
            trackers.append(result)
            return result
        with self.temporary_directory() as directory:
            with patch.object(Path, 'open', track_open), \
                 patch('hindsight_grpo.train.Tracker', side_effect=tracker), \
                 patch('hindsight_grpo.train.load_model', side_effect=RuntimeError('model load failure')):
                with self.assertRaisesRegex(RuntimeError, 'model load failure'):
                    self.create(directory)
            self.assertTrue(handles and all(handle.closed for handle in handles))
            self.assertEqual(trackers[0].finished_exit_code, 1)

    def test_shutdown_error_preserves_primary_error_and_releases_lock(self):
        with self.temporary_directory() as directory:
            experiment = self.create(directory)
            experiment.tracker.finish = Mock(side_effect=RuntimeError('tracker shutdown failure'))
            with self.assertRaisesRegex(ValueError, 'primary training failure') as caught:
                with experiment:
                    raise ValueError('primary training failure')
            self.assertTrue(experiment.lock.closed)
            self.assertIn('RuntimeError', caught.exception.__notes__[0])

    def test_partial_accumulation_resume_matches_uninterrupted_all_arms(self):
        for arm in ('root', 'entropy_cf', 'r_cf', 'gradient_ce_cf'):
            with self.subTest(arm=arm), self.temporary_directory() as directory:
                direct = self.create(Path(directory) / 'direct', arm)
                self.assertEqual(direct.run(), 0)
                expected = {name: value.clone() for name, value in direct.model.state_dict().items()}
                self.close(direct)
                interrupted = self.create(Path(directory) / 'resumed', arm)
                checkpoint = interrupted.checkpoint
                def stop_after_two(*args, **kwargs):
                    checkpoint(*args, **kwargs)
                    if interrupted.progress['cursor'] == 2:
                        interrupted.signals.requested = True
                interrupted.checkpoint = stop_after_two
                self.assertEqual(interrupted.run(), 75)
                self.assertEqual(interrupted.progress['accumulation_count'], 2)
                if arm != 'root':
                    self.assertEqual(interrupted.progress['cf_valid_count'], 2)
                    self.assertTrue(interrupted.progress['cf_gradients'])
                self.close(interrupted)
                resumed = self.create(Path(directory) / 'resumed', arm)
                self.assertEqual(resumed.progress['cursor'], 2)
                self.assertEqual(resumed.run(), 0)
                self.assertEqual(resumed.progress['prompts_processed'], 4)
                for name, value in resumed.model.state_dict().items():
                    torch.testing.assert_close(expected[name], value, rtol=0, atol=0)
                self.close(resumed)

    def test_cf_gradient_averages_only_valid_states(self):
        # Inject one known auxiliary-gradient sum for a four-prompt batch. It
        # must contribute /1 valid state, rather than /4 training prompts.
        with self.temporary_directory() as directory:
            experiment = self.create(directory, 'entropy_cf')
            experiment.config['max_grad_norm'] = 100.
            initial = experiment.model.adapter.weight.detach().clone()
            experiment.optimizer = torch.optim.SGD(experiment.parameters, lr=.1)
            def collect(row, denominator):
                if experiment.progress['cursor'] == 0:
                    experiment.progress['cf_gradients']['adapter.weight'] = torch.ones_like(initial)
                    experiment.progress['cf_valid_count'] = 1
                    return {'loss/root': 0., 'loss/counterfactual': 2., 'loss/total': .5}
                return {'loss/root': 0., 'loss/total': 0.}
            experiment.collect_prompt = collect
            self.assertEqual(experiment.run(), 0)
            torch.testing.assert_close(experiment.model.adapter.weight, initial - .1)
            metrics = experiment.tracker.logs[-1][0]
            self.assertEqual(metrics['loss/counterfactual'], 2.)
            self.assertEqual(metrics['loss/total'], .5)
            self.close(experiment)

    def test_three_epochs_flush_short_batches_without_repeating_prompts(self):
        # Six examples exercise a four-prompt update plus a two-prompt tail,
        # matching the official split's remainder without 1,122 real rollouts.
        with self.temporary_directory() as directory:
            experiment = self.create(directory, verify=False)
            experiment.config['epochs'] = 3
            experiment.config['validation_every_updates'] = 999
            experiment.data['train'] = [{'task_id': i} for i in range(6)]
            experiment.progress['baseline_done'] = True
            seen = []
            def collect(row, denominator):
                seen.append((experiment.progress['epoch'], row['task_id'], denominator))
                loss = experiment.model.adapter.weight.square().mean()
                (loss / denominator).backward()
                return {'loss/root': float(loss.detach())}
            experiment.collect_prompt = collect
            experiment.evaluate = lambda *args: {'accuracy': .5, 'pass_at_1': .5}
            with patch('hindsight_grpo.plots.make_plots', return_value=[]):
                self.assertEqual(experiment.run(), 0)
            self.assertEqual(experiment.progress['prompts_processed'], 18)
            self.assertEqual(experiment.progress['update'], 6)
            self.assertEqual(experiment.progress['accumulation_count'], 0)
            for epoch in range(3):
                rows = [(task, denominator) for e, task, denominator in seen if e == epoch]
                self.assertEqual(sorted(task for task, _ in rows), list(range(6)))
                self.assertEqual([d for _, d in rows], [4, 4, 4, 4, 2, 2])
            self.close(experiment)

    def test_signal_during_best_evaluation_preserves_final_weights(self):
        with self.temporary_directory() as directory:
            experiment = self.create(directory, verify=False)
            experiment.progress.update(epoch=experiment.config['epochs'], update=1,
                baseline_done=True, validated_update=1, final_test_done=True,
                best_validation=.5)
            final = experiment.model.adapter.weight.detach().clone()
            experiment.checkpoint()
            with torch.no_grad():
                experiment.model.adapter.weight.add_(7.)
            experiment.checkpoint('best')
            load_checkpoint(Path(directory) / 'checkpoints/latest.pt', experiment.model, experiment.optimizer)
            def stop(label, split):
                self.assertEqual(label, 'best')
                raise StopRequested()
            experiment.evaluate = stop
            self.assertEqual(experiment.run(), 75)
            torch.testing.assert_close(experiment.model.adapter.weight, final, rtol=0, atol=0)
            restored = TinyAdapterModel()
            load_checkpoint(Path(directory) / 'checkpoints/latest.pt', restored)
            torch.testing.assert_close(restored.adapter.weight, final, rtol=0, atol=0)
            self.close(experiment)

    def test_verification_resume_after_update_does_not_consume_extra_prompt(self):
        with self.temporary_directory() as directory:
            original = self.create(directory)
            self.assertEqual(original.run(), 0)
            self.close(original)
            marker = Path(directory) / 'verification_passed.json'
            marker.unlink()
            resumed = self.create(directory)
            resumed.collect_prompt = lambda *args: self.fail('verification reran training')
            self.assertEqual(resumed.run(), 0)
            self.assertEqual(resumed.progress['prompts_processed'], 4)
            self.assertTrue(marker.exists())
            self.close(resumed)

    def test_complete_training_evaluates_baseline_final_best_and_uploads_plots(self):
        with self.temporary_directory() as directory:
            experiment = self.create(directory, verify=False)
            plots = Mock(return_value=[])
            with patch.dict('sys.modules', {'hindsight_grpo.plots': SimpleNamespace(make_plots=plots)}):
                self.assertEqual(experiment.run(), 0)
            self.assertTrue(experiment.progress['complete'])
            self.assertTrue(experiment.progress['baseline_done'])
            self.assertTrue(experiment.progress['final_test_done'])
            self.assertTrue(experiment.progress['best_test_done'])
            self.assertEqual(experiment.progress['prompts_processed'], 4)
            plots.assert_called_once()
            labels = {(record['label'], record['split'])
                      for kind, record, step in experiment.tracker.records if kind == 'evaluation'}
            self.assertEqual(labels, {('pretrained', 'validation'), ('pretrained', 'test'),
                ('update-1', 'validation'), ('final', 'test'), ('best', 'test')})
            for metrics, step in experiment.tracker.logs:
                if 'validation/accuracy' in metrics:
                    self.assertGreater(metrics['compute/gpu_hours'], 0.)
            self.assertIn('cost/evaluation_seconds', experiment.tracker.logs[-1][0])
            self.close(experiment)

    def test_failure_record_redacts_credentials_and_raises(self):
        with self.temporary_directory() as directory:
            experiment = self.create(directory)
            def fail(*args):
                raise RuntimeError('synthetic failure containing dummy-not-real')
            experiment.collect_prompt = fail
            with patch.dict('os.environ', {'WANDB_API_KEY': 'dummy-not-real'}):
                with self.assertRaises(RuntimeError):
                    experiment.run()
            errors = [record for kind, record, step in experiment.tracker.records if kind == 'failure']
            self.assertEqual(len(errors), 1)
            self.assertNotIn('dummy-not-real', errors[0]['message'])
            self.assertIn('[redacted]', errors[0]['message'])
            self.close(experiment)


if __name__ == '__main__':
    unittest.main()
