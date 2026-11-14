"""CPU transaction and frozen-policy integration tests, with no program execution."""
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from d1_exclusion.config import load_config
from d1_exclusion.trainer import Trainer


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 5)
        self.base = torch.nn.Linear(5, 7)
        self.adapter = torch.nn.Linear(5, 7, bias=False)
        self.embedding.requires_grad_(False)
        self.base.requires_grad_(False)
        torch.nn.init.zeros_(self.adapter.weight)
        self.config = SimpleNamespace(eos_token_id=None, pad_token_id=None)
        self.disabled = False

    @contextmanager
    def disable_adapter(self):
        assert not self.disabled
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        hidden = hidden + hidden.mean(1, keepdim=True)
        logits = self.base(hidden)
        if not self.disabled:
            logits = logits + self.adapter(hidden)
        logits = logits.clone()
        logits[..., 0] = -100
        return SimpleNamespace(logits=logits)


class Tokenizer:
    mask_token_id = 0
    eos_token_id = None
    pad_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return "synthetic prompt"

    def encode(self, text, **kwargs):
        return [1, 2]

    def decode(self, ids, **kwargs):
        return ",".join(map(str, ids))


@dataclass
class Reward:
    reward: float
    partial: float
    syntax_valid: bool = True
    error: str = ""
    timeout: bool = False

    def to_dict(self):
        return asdict(self)


class RewardRunner:
    def evaluate(self, text, row):
        result = float(sum(map(int, text.split(","))) % 2)
        return Reward(result, result)


@pytest.fixture(autouse=True)
def tiny_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with patch("d1_exclusion.trainer.implementation_token_mask", side_effect=lambda ids, tokenizer: [True] * len(ids)):
        yield
    torch.set_num_threads(threads)


def make(directory, arm="d1_root", resume=False):
    names = {"d1_root": "root", "entropy_no_ban": "no_ban", "entropy_soft_ban": "soft_ban", "entropy_hard_ban": "hard_ban"}
    path = Path(__file__).parents[2] / f"configs/d1_exclusion/{names[arm]}.json"
    config = load_config(path)
    config.update(training_tasks=6, generation_length=4, denoising_steps=4, block_size=2,
                  learning_rate=0.002)
    torch.manual_seed(46)
    model = Model()
    rows = [{"task_id": i, "text": str(i), "test_list": []} for i in range(6)]
    return Trainer(config, directory, model, Tokenizer(), RewardRunner(), rows, resume=resume)


@pytest.mark.parametrize("arm", ["d1_root", "entropy_no_ban", "entropy_soft_ban", "entropy_hard_ban"])
def test_frozen_collection_twelve_steps_final_two_and_exact_restarts(tmp_path, arm):
    with make(tmp_path / "direct", arm) as direct:
        assert direct.optimizer.param_groups[0]["lr"] == 0
        assert direct.scheduler.last_epoch == 0
        root_before = copy.deepcopy(direct.model.state_dict())
        collections = []
        original = direct.collect_prompt
        def count(row):
            result = original(row)
            collections.append(copy.deepcopy(result))
            return result
        direct.collect_prompt = count
        assert direct.run() == 0
        expected_weights = copy.deepcopy(direct.model.state_dict())
        expected_optimizer = copy.deepcopy(direct.optimizer.state_dict())
        expected_scheduler = copy.deepcopy(direct.scheduler.state_dict())
        assert len(collections) == 6  # No re-generation during twelve inner updates.
        assert direct.progress["outer_collection"] == 2
        assert direct.progress["optimizer_steps"] == 24
        assert direct.progress["applied_prompts"] == direct.progress["cursor"] == 6
        assert not torch.equal(root_before["adapter.weight"], expected_weights["adapter.weight"])
        assert torch.equal(root_before["embedding.weight"], expected_weights["embedding.weight"])
        assert direct.progress["last_metrics"]["prompts_in_collection"] == 2
        assert direct.progress["last_metrics"]["valid_response_tokens"] == 2 * 8 * 4
        import json
        first_step = json.loads((direct.run_dir / "steps/000001.json").read_text())
        assert first_step["learning_rate"] == 0
        assert first_step["next_learning_rate"] == direct.config["learning_rate"]
        assert (direct.run_dir / "completed.json").exists()
        for record in collections:
            assert sum(record["advantages"]) == pytest.approx(0)
            assert len(record["roots"]) == 8
    # Interrupt mid-collection, mid-inner loop, after final inner step and at a boundary.
    cases = [("prompt", 2), ("optimizer", 5), ("optimizer", 12), ("collection", 1), ("optimizer", 14)]
    for event, threshold in cases:
        directory = tmp_path / f"{event}-{threshold}"
        with make(directory, arm) as stopped:
            def interrupt(trainer, transaction):
                key = {"prompt": "collected_prompts", "optimizer": "optimizer_steps", "collection": "outer_collection"}[event]
                if transaction == event and trainer.progress[key] == threshold:
                    trainer.signals.requested = True
            assert stopped.run(interrupt) == 75
            assert not (directory / "completed.json").exists()
        with make(directory, arm, resume=True) as resumed:
            assert resumed.run() == 0
            assert resumed.progress["optimizer_steps"] == 24
            for name, weight in resumed.model.state_dict().items():
                torch.testing.assert_close(weight, expected_weights[name], rtol=0, atol=0)
            for index, values in resumed.optimizer.state_dict()["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, expected_optimizer["state"][index][name], rtol=0, atol=0)
            assert resumed.scheduler.state_dict() == expected_scheduler


def test_no_overwrite_protocol_mismatch_and_locking(tmp_path):
    with make(tmp_path) as trainer:
        with pytest.raises(BlockingIOError):
            make(tmp_path, resume=True)
    with pytest.raises(FileExistsError):
        make(tmp_path)
    with pytest.raises(ValueError, match="configuration differs"):
        make(tmp_path, "entropy_soft_ban", resume=True)


def test_no_alternative_skips_only_auxiliary_training(tmp_path):
    from d1_exclusion.counterfactual import estimate_conditional_values
    with make(tmp_path, "entropy_no_ban") as trainer:
        original = trainer.collect_prompt
        def no_alternatives(row):
            result = original(row)
            cf = result["cf"]
            cf.update(estimate_conditional_values([0] * 4, [0] * 8, [1] * 8, 1, -0.01))
            return result
        trainer.collect_prompt = no_alternatives
        assert trainer.run() == 0
        assert trainer.progress["optimizer_steps"] == 24
        assert trainer.progress["last_metrics"]["cf/valid_states"] == 0
        assert trainer.progress["last_metrics"]["loss/cf"] == 0


def test_atomic_failed_step_save_resumes_previous_durable_model(tmp_path):
    with make(tmp_path / "direct") as direct:
        direct.run()
        expected = copy.deepcopy(direct.model.state_dict())
    directory = tmp_path / "failed"
    with make(directory) as trainer:
        real_save = trainer.checkpoint
        def fail_save(*args, **kwargs):
            if trainer.progress["optimizer_steps"] == 3:
                raise OSError("simulated save failure after in-memory step")
            return real_save(*args, **kwargs)
        trainer.checkpoint = fail_save
        with pytest.raises(OSError):
            trainer.run()
    with make(directory, resume=True) as resumed:
        assert resumed.progress["optimizer_steps"] == 2
        assert resumed.run() == 0
        for name, weight in resumed.model.state_dict().items():
            torch.testing.assert_close(weight, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["entropy_no_ban", "entropy_soft_ban", "entropy_hard_ban"])
@pytest.mark.parametrize("contrast", [-1., 0., 1.])
def test_reference_reward_never_enters_conditional_estimates_or_changes_credit(tmp_path, arm, contrast):
    """Force known fresh-branch values while using real tiny-model continuations."""
    results = []
    for reference_reward in (0., 1.):
        with make(tmp_path / str(reference_reward), arm) as trainer:
            calls = []
            keep_value = float(contrast <= 0)
            alternative_value = float(contrast >= 0)
            if contrast == 0:
                alternative_value = keep_value

            def score(trajectory, row):
                if len(calls) < 8:
                    value = reference_reward
                elif trajectory.metadata["intervention"] == "keep":
                    value = keep_value
                else:
                    value = alternative_value if trajectory.metadata["changed"] else keep_value
                calls.append((trajectory.seed, value))
                return Reward(value, value).to_dict()

            trainer._score = score
            record = trainer.collect_prompt(trainer.rows[0])
            cf = record["cf"]
            assert record["advantages"] == [0.] * 8
            assert cf["root_group_type"] == ("all_successful" if reference_reward else "all_failed")
            assert cf["successful_roots"] == int(reference_reward) * 8
            assert cf["reference_reward"] == reference_reward
            assert len(calls) == 8 + 4 + 8
            assert cf["keep_sample_count"] == len(cf["keep_rewards"]) == 4
            assert cf["proposal_sample_count"] == len(cf["proposal_rewards"]) == 8
            assert cf["valid"] and cf["n_alt"] > 0
            assert cf["n_alt"] + cf["n_keep"] == 8
            assert cf["q_keep"] == keep_value
            assert cf["q_alt"] == alternative_value
            assert cf["delta"] == contrast
            assert cf["keep_denominator"] == 4 + cf["n_keep"]
            assert all(reward["reward"] == keep_value for reward in cf["keep_rewards"])
            # Every branch is fresh; the outcome-conditioned root seed is absent.
            root_seeds = {root["seed"] for root in record["roots"]}
            branch_seeds = cf["keep_seeds"] + cf["proposal_seeds"]
            assert len(set(branch_seeds)) == 12
            assert not root_seeds & set(branch_seeds)
            assert calls[8:12] == list(zip(cf["keep_seeds"], [keep_value] * 4))
            if contrast > 0:
                assert cf["target_shift"] < 0
            elif contrast < 0:
                assert cf["target_shift"] > 0
            else:
                assert cf["target_shift"] == pytest.approx(0, abs=1e-15)
            assert cf["target_probability"] == pytest.approx(torch.tensor(cf["target_logprob"]).exp().item())
            results.append((cf["delta"], cf["target_logprob"], cf["q_keep"], cf["q_alt"]))
            # Homogeneous rewards give zero root advantages but can still yield
            # nonzero auxiliary loss and a real selected-state gradient.
            trainer.collection = [record]
            target = cf["target_logprob"]
            trainer.inner_step()
            assert cf["target_logprob"] == target
            assert trainer.progress["last_metrics"]["loss/pg"] == 0
            if contrast:
                assert trainer.progress["last_metrics"]["loss/cf"] > 0
                assert trainer.progress["last_metrics"]["gradient_norm"] > 0
    assert results[0] == results[1]


def test_every_counterfactual_arm_calls_identical_stratified_selector_once(tmp_path):
    from d1_exclusion.selector import select_reference
    outcomes = [1., 0., 0., 0., 0., 0., 0., 0.]
    choices = []
    for arm in ("entropy_no_ban", "entropy_soft_ban", "entropy_hard_ban"):
        with make(tmp_path / arm, arm) as trainer:
            count = 0
            def score(trajectory, row):
                nonlocal count
                value = outcomes[count] if count < 8 else 0.
                count += 1
                return Reward(value, value).to_dict()
            trainer._score = score
            with patch("d1_exclusion.trainer.select_reference", wraps=select_reference) as selector:
                record = trainer.collect_prompt(trainer.rows[0])
            selector.assert_called_once()
            assert selector.call_args.args[0] == outcomes
            cf = record["cf"]
            assert cf["root_group_type"] == "mixed"
            assert cf["successful_roots"] == 1
            assert cf["mixed_reference_outcome_class"] == ("successful" if cf["reference_reward"] else "failed")
            assert cf["reference_selection_method"] == trainer.config["reference_selection"]
            assert select_reference(outcomes, cf["reference_selection_seed"])["reference_index"] == cf["reference_index"]
            choices.append((cf["reference_index"], cf["selected_index"], cf["reference_selection_seed"]))
    assert choices[0] == choices[1] == choices[2]


def test_root_only_does_not_select_reference_or_draw_conditional_branches(tmp_path):
    with make(tmp_path) as trainer, \
            patch("d1_exclusion.trainer.select_reference", side_effect=AssertionError("root must not select")), \
            patch.object(trainer.sampler, "counterfactual_groups", side_effect=AssertionError("root must not branch")):
        record = trainer.collect_prompt(trainer.rows[0])
        assert record["cf"] is None
        assert len(record["roots"]) == len(record["rewards"]) == 8


def test_ineligible_reference_still_logs_outcome_metadata_and_missing_fields(tmp_path):
    with make(tmp_path, "entropy_hard_ban") as trainer, \
            patch("d1_exclusion.trainer.implementation_token_mask", side_effect=lambda ids, tokenizer: [False] * len(ids)):
        record = trainer.collect_prompt(trainer.rows[0])
        cf = record["cf"]
        assert cf["reference_reward"] in (0, 1)
        assert cf["root_group_type"] in ("all_successful", "all_failed", "mixed")
        assert not cf["valid"] and cf["skip_reason"] == "no_eligible_implementation_commitment"
        for key in ("selected_index", "old_keep_probability", "selected_progress", "n_alt", "n_keep",
                    "q_keep", "q_alt", "delta", "target_probability", "target_shift"):
            assert cf[key] is None
        assert cf["keep_sample_count"] == cf["proposal_sample_count"] == 0


def test_amendment_marker_prevents_silent_resume_of_uniform_selection(tmp_path):
    import json
    with make(tmp_path, "entropy_soft_ban"):
        pass
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    del config["reference_selection"]
    path.write_text(json.dumps(config))
    checkpoint = (tmp_path / "checkpoints/latest.pt").read_bytes()
    with pytest.raises(ValueError, match="configuration differs"):
        make(tmp_path, "entropy_soft_ban", resume=True)
    assert (tmp_path / "checkpoints/latest.pt").read_bytes() == checkpoint
