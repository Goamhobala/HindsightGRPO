from pathlib import Path
from unittest.mock import patch

import pytest

from hindsight_grpo.data import (
    DATASET_REVISION, EXPECTED_SPLIT_SIZES, extract_code,
    implementation_span, implementation_token_mask, prompt_for,
)
from hindsight_grpo.rewards import RewardRunner, SandboxUnavailable, find_bubblewrap


def test_official_configuration_and_no_solution_in_prompt():
    assert EXPECTED_SPLIT_SIZES == {"train": 374, "validation": 90, "test": 500, "prompt": 10}
    assert len(DATASET_REVISION) == 40
    prompt = prompt_for({"text": "Double x.", "test_list": ["assert f(2) == 4"],
                         "code": "secret_reference_solution"})
    assert "assert f(2) == 4" in prompt
    assert "secret_reference_solution" not in prompt


def test_extract_code_preserves_invalid_code_and_excludes_tests():
    response = "Here is the answer:\n```python\ndef f(x):\n    return x + 1\n\nassert f(2) == 3\n```\nExplanation"
    assert extract_code(response) == "def f(x):\n    return x + 1"
    assert extract_code("def f(:\n    broken") == "def f(:\n    broken"
    assert extract_code("This is only explanatory prose.") == ""
    assert extract_code("def test_value(x):\n    return x\n") == "def test_value(x):\n    return x"


def test_extract_code_unfenced_excludes_prose_and_test_functions():
    text = "The solution follows.\ndef f(x):\n    return x\n\ndef test_f():\n    assert f(1) == 1\n"
    assert extract_code(text) == "def f(x):\n    return x"
    start, end = implementation_span(text)
    assert text[start:end] == extract_code(text)


class CharacterTokenizer:
    all_special_ids = [0]
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_token_eligibility_excludes_fences_prose_tests_and_stop():
    text = "```python\ndef f(x):\n    return x\nassert f(1) == 1\n```"
    ids = [ord(c) for c in text] + [0]
    mask = implementation_token_mask(ids, CharacterTokenizer())
    selected = "".join(c for c, eligible in zip(text, mask) if eligible)
    assert selected == "def f(x):\n    return x"
    assert not mask[-1]
    # Indentation remains part of implementation and commitment progress.
    assert mask[text.index("    return")]


def test_token_alignment_failure_is_conservative():
    tokenizer = CharacterTokenizer()
    tokenizer.decode = lambda ids, **kwargs: "x"
    with pytest.warns(RuntimeWarning, match="token boundaries"):
        assert implementation_token_mask([100, 101], tokenizer) == [False, False]


def test_sandbox_fail_closed_on_non_linux():
    runner = RewardRunner(require_slurm=False)
    with patch("hindsight_grpo.rewards.platform.system", return_value="Darwin"):
        with pytest.raises(SandboxUnavailable, match="Linux bubblewrap"):
            runner.check_available()


def test_generated_code_requires_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    runner = RewardRunner()
    with pytest.raises(SandboxUnavailable, match="SLURM"):
        runner.evaluate("def f():\n    return 1", {"test_list": ["assert f() == 1"]})


def test_sandbox_payload_malformed_output_cannot_count_as_pass():
    runner = RewardRunner(require_slurm=False)
    runner._verified = True
    with patch.object(runner, "_run", return_value={"error": "invalid_worker_result", "tests_passed": 0}):
        result = runner.evaluate("def f(x):\n    return x", {"test_list": ["assert f(1) == 1"]})
    assert result.reward == 0 and result.partial == 0 and result.syntax_valid


def test_partial_fraction_and_error_reporting():
    runner = RewardRunner(require_slurm=False)
    runner._verified = True
    fake = {"tests_passed": 1, "test_errors": [None, "AssertionError"], "error": "AssertionError"}
    with patch.object(runner, "_run", return_value=fake):
        result = runner.evaluate("def f(x):\n    return x", {"test_list": ["assert f(1) == 1", "assert f(2) == 3"]})
    assert result.reward == 0 and result.partial == 0.5
    assert result.test_errors == (None, "AssertionError")
    assert result.to_dict()["error"] == "AssertionError"


def test_actual_sandbox_reward_and_simple_harness_mutations():
    import os
    import platform
    if platform.system() != "Linux" or find_bubblewrap() is None or not os.environ.get("SLURM_JOB_ID"):
        pytest.skip("Linux bubblewrap integration requires a SLURM compute-node environment")
    runner = RewardRunner(require_slurm=False)
    runner.check_available()
    row = {"test_list": ["assert f(1) == 2", "assert f(2) == 3"]}
    assert runner.evaluate("def f(x):\n    return x + 1", row).reward == 1
    assert runner.evaluate("import sys\nsys.exit(0)\ndef f(x):\n    return x + 1", row).reward == 0
    malicious = "import builtins\nbuiltins.exec = lambda *args: None\ndef f(x):\n    return 0"
    result = runner.evaluate(malicious, row)
    assert result.reward == 0 and result.tests_passed == 0


def test_sandbox_command_enforces_mount_and_namespace_limits():
    runner = RewardRunner(require_slurm=False)
    runner.bwrap = "/usr/bin/bwrap"
    runner.python = Path("/usr/bin/python3")
    with patch("hindsight_grpo.rewards.platform.system", return_value="Linux"):
        command = runner._command(Path("/tmp/fixed-input.json"))
    assert "--clearenv" in command and "--disable-userns" in command
    assert "--unshare-user" in command and "--unshare-all" in command
    assert command[command.index("--size") + 1] == str(16 * 1024 * 1024)
    readonly = [command[i + 1] for i, value in enumerate(command) if value == "--remount-ro"]
    assert {"/", "/dev", "/proc"}.issubset(readonly)
    assert "WANDB_API_KEY" not in command
