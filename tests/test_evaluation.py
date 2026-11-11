"""Evaluation combinatorics and per-problem restart behavior without model/code execution."""
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight_grpo.evaluation import BudgetExpired, StopRequested, evaluate, pass_at_k


@pytest.mark.parametrize("n", range(1, 8))
def test_pass_at_k_matches_enumerated_sampling_without_replacement(n):
    for correct in range(n + 1):
        outcomes = [True] * correct + [False] * (n - correct)
        for k in range(1, n + 1):
            subsets = list(itertools.combinations(outcomes, k))
            exact = sum(any(subset) for subset in subsets) / len(subsets)
            assert pass_at_k(n, correct, k) == pytest.approx(exact)


@pytest.mark.parametrize("arguments", [(0, 0, 1), (3, -1, 1), (3, 4, 1), (3, 1, 0), (3, 1, 4)])
def test_pass_at_k_rejects_invalid_counts(arguments):
    with pytest.raises(ValueError):
        pass_at_k(*arguments)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, **kwargs):
        return [len(text)]

    def decode(self, ids, **kwargs):
        return f"{ids[0]}:{ids[1]}"


class FakeTrajectory:
    def __init__(self, task_id, sample, seed):
        self.generated_ids = [task_id, sample]
        self.seed = seed

    def to_dict(self):
        return {"generated_ids": self.generated_ids, "seed": self.seed}


class FakeSampler:
    def __init__(self, callback=None):
        self.calls = []
        self.counts = {}
        self.callback = callback

    def sample(self, ids, seed, prompt_id):
        task_id = int(prompt_id)
        sample = self.counts.get(task_id, 0)
        self.counts[task_id] = sample + 1
        self.calls.append((task_id, sample, seed))
        if self.callback:
            self.callback(task_id, sample)
        return FakeTrajectory(task_id, sample, seed)


class FakeRewardRunner:
    def __init__(self):
        self.calls = []

    def evaluate(self, text, row):
        self.calls.append((text, row["task_id"]))
        task_id, sample = map(int, text.split(":"))
        reward = float(task_id == 10 and sample == 0)
        result = {"reward": reward, "partial": 1.0 if reward else 0.5,
                  "syntax_valid": True, "timeout": False,
                  "error": None if reward else "AssertionError"}
        return SimpleNamespace(to_dict=lambda: result)


class FakeTracker:
    arm = "root"

    def __init__(self):
        self.records = []
        self.logs = []
        self.artifacts = []
        self.flushes = 0

    def record(self, *args):
        self.records.append(args)

    def log(self, *args):
        self.logs.append(args)

    def artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))

    def flush_records(self):
        self.flushes += 1


ROWS = [{"task_id": task, "text": f"Task {task}", "test_list": ["assert True"]}
        for task in (10, 20)]
CONFIG = {"evaluation_samples": 2, "seed": 19}


def run_evaluation(tmp_path, sampler=None, reward_runner=None, tracker=None, signals=None, split="validation", checkpoint_id=None):
    return evaluate(sampler or FakeSampler(), FakeTokenizer(), reward_runner or FakeRewardRunner(),
                    ROWS, split, "update-7", CONFIG, tmp_path, tracker or FakeTracker(), 7, signals, checkpoint_id)


def test_evaluation_metrics_and_cached_resume_do_not_repeat_problems(tmp_path):
    first_sampler, first_rewards, first_tracker = FakeSampler(), FakeRewardRunner(), FakeTracker()
    metrics = run_evaluation(tmp_path, first_sampler, first_rewards, first_tracker)
    assert metrics["accuracy"] == pytest.approx(0.25)
    assert metrics["partial"] == pytest.approx(0.625)
    assert metrics["syntax_valid"] == 1.0
    assert metrics["execution_errors"] == pytest.approx(0.75)
    assert metrics["pass_at_1"] == pytest.approx(0.25)
    assert metrics["pass_at_2"] == pytest.approx(0.5)
    assert len(first_sampler.calls) == len(first_rewards.calls) == 4
    assert len(first_tracker.records) == 2
    assert any("eval/update-7/validation/accuracy" in entry[0] for entry in first_tracker.logs)
    cache = tmp_path / "evaluation" / "update-7" / "validation"
    original = {task: (cache / f"{task}.json").read_bytes() for task in (10, 20)}
    second_sampler, second_rewards, second_tracker = FakeSampler(), FakeRewardRunner(), FakeTracker()
    resumed = run_evaluation(tmp_path, second_sampler, second_rewards, second_tracker)
    for key in metrics.keys() - {"seconds"}:
        assert resumed[key] == metrics[key]
    assert second_sampler.calls == second_rewards.calls == second_tracker.records == []
    assert second_tracker.flushes == 1 and len(second_tracker.artifacts) == 1
    assert all((cache / f"{task}.json").read_bytes() == original[task] for task in original)


def test_interrupted_evaluation_resumes_only_missing_problems(tmp_path):
    signals = SimpleNamespace(requested=False)

    def interrupt_after_first_problem(task_id, sample):
        if task_id == 10 and sample == 1:
            signals.requested = True

    sampler = FakeSampler(interrupt_after_first_problem)
    with pytest.raises(StopRequested):
        run_evaluation(tmp_path, sampler=sampler, signals=signals)
    cache = tmp_path / "evaluation" / "update-7" / "validation"
    assert (cache / "10.json").exists()
    assert not (cache / "20.json").exists()
    assert not (cache / "summary.json").exists()
    original = (cache / "10.json").read_bytes()
    signals.requested = False
    resumed_sampler = FakeSampler()
    resumed = run_evaluation(tmp_path, sampler=resumed_sampler, signals=signals)
    assert [task for task, _, _ in resumed_sampler.calls] == [20, 20]
    assert (cache / "10.json").read_bytes() == original
    assert resumed["problems"] == 2 and (cache / "summary.json").exists()


def test_test_split_scores_do_not_write_validation_selection_metrics(tmp_path):
    tracker = FakeTracker()
    run_evaluation(tmp_path, tracker=tracker, split="test")
    assert not any("validation/accuracy" in entry[0] for entry in tracker.logs)
    assert any("eval/update-7/test/accuracy" in entry[0] for entry in tracker.logs)


def test_already_requested_stop_generates_nothing(tmp_path):
    sampler = FakeSampler()
    with pytest.raises(StopRequested):
        run_evaluation(tmp_path, sampler=sampler, signals=SimpleNamespace(requested=True))
    assert sampler.calls == []


@pytest.mark.parametrize("mutation", ["samples", "task_id", "split", "provenance", "rewards", "trajectories"])
def test_mismatched_cached_evaluation_is_rejected(tmp_path, mutation):
    import json
    run_evaluation(tmp_path)
    path = tmp_path / "evaluation" / "update-7" / "validation" / "10.json"
    record = json.loads(path.read_text())
    if mutation in {"rewards", "trajectories"}:
        record[mutation] = record[mutation][:-1]
    elif mutation == "provenance":
        record[mutation]["config_digest"] = "different-protocol"
    else:
        record[mutation] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provenance or sample-count"):
        run_evaluation(tmp_path)


def test_evaluation_cache_rejects_different_checkpoint_identity(tmp_path):
    run_evaluation(tmp_path, checkpoint_id="best-update-7")
    with pytest.raises(ValueError, match="provenance or sample-count"):
        run_evaluation(tmp_path, checkpoint_id="best-update-8")


def test_pilot_reuses_pretrained_subset_without_overwriting_full_baseline(tmp_path):
    evaluate(FakeSampler(), FakeTokenizer(), FakeRewardRunner(), ROWS, 'test', 'pretrained',
             CONFIG, tmp_path, FakeTracker(), 0)
    original = tmp_path / 'evaluation/pretrained/test/summary.json'
    before = original.read_bytes()
    sampler, rewards, tracker = FakeSampler(), FakeRewardRunner(), FakeTracker()
    result = evaluate(sampler, FakeTokenizer(), rewards, ROWS[:1], 'test', 'pilot-4-pretrained',
                      CONFIG, tmp_path, tracker, 0, checkpoint_id='pretrained', cache_source='pretrained')
    assert sampler.calls == rewards.calls == []
    assert result['problems'] == 1
    assert result['accuracy'] == .5
    assert original.read_bytes() == before
    assert len(tracker.records) == 1
    assert tracker.records[0][1]['label'] == 'pilot-4-pretrained'


def test_pilot_reuse_rejects_wrong_pretrained_cache_provenance(tmp_path):
    import json
    evaluate(FakeSampler(), FakeTokenizer(), FakeRewardRunner(), ROWS, 'test', 'pretrained',
             CONFIG, tmp_path, FakeTracker(), 0)
    path = tmp_path / 'evaluation/pretrained/test/10.json'
    record = json.loads(path.read_text())
    record['provenance']['checkpoint_id'] = 'different-checkpoint'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='provenance'):
        evaluate(FakeSampler(), FakeTokenizer(), FakeRewardRunner(), ROWS, 'test', 'pilot-4-pretrained',
                 CONFIG, tmp_path, FakeTracker(), 0, checkpoint_id='pretrained', cache_source='pretrained')


def test_evaluation_reports_sample_and_problem_progress(tmp_path):
    progress = []
    evaluate(FakeSampler(), FakeTokenizer(), FakeRewardRunner(), ROWS, 'validation', 'update-7',
             CONFIG, tmp_path, FakeTracker(), 7, progress_callback=progress.append)
    assert progress[0]['progress/problems_completed'] == 0
    assert progress[-1]['progress/problems_completed'] == 2
    assert progress[-1]['progress/problems_total'] == 2
    assert any(p['progress/samples_completed'] == 1 for p in progress)


def test_budget_deadline_stops_without_generating_a_problem(tmp_path):
    sampler = FakeSampler()
    with pytest.raises(BudgetExpired):
        evaluate(sampler, FakeTokenizer(), FakeRewardRunner(), ROWS, 'test', 'final',
                 CONFIG, tmp_path, FakeTracker(), 7, deadline_unix=1)
    assert sampler.calls == []
