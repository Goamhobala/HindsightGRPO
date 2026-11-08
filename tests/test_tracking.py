import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hindsight_grpo.tracking import Tracker, _safe


class FakeArtifact:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.files = []
    def add_file(self, path, name):
        self.files.append((path, name))
    def add_dir(self, path, name):
        self.files.append((path, name))


class FakeRun:
    offline = False
    url = 'https://wandb.ai/unit/project/runs/example'
    project = 'project'
    entity = 'unit'
    def __init__(self):
        self.logs = []
        self.artifacts = []
        self.summary = {}
    def define_metric(self, *args, **kwargs):
        pass
    def log(self, value):
        self.logs.append(value)
    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))
    def finish(self, **kwargs):
        self.finished = kwargs


class TrackingTests(unittest.TestCase):
    def test_missing_key_prevents_run(self):
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'WANDB_API_KEY'):
                Tracker(temporary, 'root', {})

    def test_offline_mode_is_rejected(self):
        with patch.dict(os.environ, {'WANDB_API_KEY': 'dummy-not-real', 'WANDB_MODE': 'offline'}, clear=True):
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(RuntimeError, 'WANDB_MODE'):
                    Tracker(temporary, 'root', {})

    def test_resumption_artifacts_metrics_and_secret_hygiene(self):
        calls = []
        runs = []
        def init(**kwargs):
            calls.append(kwargs)
            run = FakeRun()
            runs.append(run)
            return run
        fake = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs, Artifact=FakeArtifact)
        with patch.dict(sys.modules, {'wandb': fake}):
            with patch.dict(os.environ, {'WANDB_API_KEY': 'dummy-not-real', 'WANDB_PROJECT': 'override'}, clear=True):
                with tempfile.TemporaryDirectory() as temporary:
                    tracker = Tracker(temporary, 'r_cf', {'wandb_project': 'config-project'}, group='experiment')
                    stable_id = tracker.run_id
                    tracker.log({'root/reward': .25}, 1)
                    tracker.log({'validation/correct': .5}, 1)
                    tracker.record('trajectories', {'prompt_id': 9, 'tokens': [1, 2]}, step=1)
                    tracker.finish()
                    self.assertEqual(calls[0]['mode'], 'online')
                    self.assertEqual(calls[0]['project'], 'override')
                    self.assertNotIn('dummy-not-real', json.dumps(calls))
                    self.assertEqual([r['update'] for r in runs[0].logs], [1, 1])
                    self.assertEqual(len(runs[0].artifacts), 2)
                    records = [json.loads(line) for line in (Path(temporary) / 'trajectories.jsonl').read_text().splitlines()]
                    self.assertEqual(records[0]['payload']['prompt_id'], 9)
                    resumed = Tracker(temporary, 'r_cf', {}, run_id=stable_id)
                    self.assertEqual(calls[1]['id'], stable_id)
                    resumed.finish()

    def test_credential_fields_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Credential'):
            _safe({'nested': {'WANDB_API_KEY': 'not-real'}})
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            _safe({'score': float('nan')})

    def test_upload_failure_closes_local_files_before_directory_cleanup(self):
        run = FakeRun()
        fake = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs,
                               Artifact=FakeArtifact)
        with patch.dict(sys.modules, {'wandb': fake}), patch.dict(os.environ,
                {'WANDB_API_KEY': 'dummy-not-real', 'WANDB_MODE': 'online'}, clear=True):
            with tempfile.TemporaryDirectory() as temporary:
                tracker = Tracker(temporary, 'root', {})
                tracker.record('records', {'task_id': 1})
                stream = tracker._files['records']
                with patch.object(tracker, 'flush_records', side_effect=RuntimeError('upload failed')):
                    with self.assertRaisesRegex(RuntimeError, 'upload failed'):
                        tracker.finish(exit_code=1)
                self.assertTrue(stream.closed)
                self.assertEqual(tracker._files, {})
                self.assertEqual(run.finished, {'exit_code': 1})


if __name__ == '__main__':
    unittest.main()
