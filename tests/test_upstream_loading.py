"""Exercise the exact pinned HF architecture without network or pretrained weights."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hindsight_grpo.model import load_model, load_pinned_architecture


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tests/fixtures/hf_mdlm_modeling_qwen2.py"
CONFIG = ROOT / "tests/fixtures/hf_mdlm_config.json"
SOURCE_SHA256 = "708c3f667f7b4eac3fd504ace32e1ee7171cfbfbbc6b6d9d95eda0353a779710"


def no_dllm_import(original):
    def checked(name, *args, **kwargs):
        if name == "dllm" or name.startswith("dllm."):
            raise AssertionError("The model runtime must not execute the guarded dllm demo")
        return original(name, *args, **kwargs)
    return checked


def test_exact_pinned_source_loads_without_demo_dependency():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    with patch("hindsight_grpo.model.hf_hub_download", return_value=str(SOURCE)), \
            patch.object(builtins, "__import__", side_effect=no_dllm_import(builtins.__import__)):
        architecture = load_pinned_architecture()
    assert architecture.source_sha256 == SOURCE_SHA256
    assert architecture.config_class.model_type == "a2d-qwen2"
    assert architecture.model_class.__name__ == "A2DQwen2LMHeadModel"


def test_real_custom_architecture_forward_lora_and_input_gradients(tmp_path):
    spec = importlib.util.spec_from_file_location("check_model_imports", ROOT / "scripts/check_model_imports.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch("hindsight_grpo.model.hf_hub_download", return_value=str(SOURCE)), \
                patch.object(builtins, "__import__", side_effect=no_dllm_import(builtins.__import__)):
            architecture = load_pinned_architecture()
            report = checker.verify_architecture(architecture, json.loads(CONFIG.read_text()))
    finally:
        torch.set_num_threads(previous_threads)
    assert report["status"] == "passed"
    assert report["source_sha256"] == SOURCE_SHA256
    assert report["tiny_logits_shape"] == [1, 5, 64]
    assert report["lora_gradient_norm"] > 0
    assert report["input_gradient_norm"] > 0
    assert report["future_token_effect_max_abs_logit"] > 1e-6
    assert report["pretrained_weights_loaded"] is False
    assert report["tiny_checkpoint_roundtrip"] is True


def test_offline_preflight_requires_cached_source_and_configuration():
    spec = importlib.util.spec_from_file_location("offline_check_model_imports", ROOT / "scripts/check_model_imports.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    architecture = object()
    with patch.object(checker, "load_pinned_architecture", return_value=architecture) as source, \
            patch.object(checker, "hf_hub_download", return_value=str(CONFIG)) as download, \
            patch.object(checker, "verify_architecture", return_value={"status": "passed"}):
        report = checker.verify_model_imports({}, offline=True)
    source.assert_called_once_with({}, local_files_only=True)
    assert download.call_args.kwargs["local_files_only"] is True
    assert download.call_args.kwargs["filename"] == "config.json"
    assert report["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_changed_source_is_rejected_before_execution(tmp_path):
    source = tmp_path / 'modeling_qwen2.py'
    source.write_text("raise AssertionError('changed code must never execute')\n")
    with patch('hindsight_grpo.model.hf_hub_download', return_value=str(source)):
        with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
            load_pinned_architecture()


def test_wrong_model_revision_is_rejected_before_download():
    with patch('hindsight_grpo.model.hf_hub_download') as download:
        with pytest.raises(ValueError, match='inspected diffusion MDLM revision'):
            load_pinned_architecture({'model_revision': 'main'})
    download.assert_not_called()


def test_training_weight_loader_uses_explicit_pinned_classes():
    # The tiny checkpoint roundtrip above checks the real HF loader. Here only
    # downloading pretrained weights is replaced, so the production loader's
    # explicit class path cannot regress to the failing Auto import scanner.
    with patch('hindsight_grpo.model.hf_hub_download', return_value=str(SOURCE)):
        architecture = load_pinned_architecture()
        config = architecture.config_class(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32,
            bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = 'eager'
        tiny = architecture.model_class(config)
        tokenizer = SimpleNamespace(mask_token_id=151665, mask_token='<|mask|>')
        with patch.object(architecture.config_class, 'from_pretrained', return_value=config), \
                patch.object(architecture.model_class, 'from_pretrained', return_value=tiny) as weights, \
                patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                patch('transformers.AutoModelForMaskedLM.from_pretrained',
                      side_effect=AssertionError('Auto dependency scanner must not run')):
            bundle = load_model(device='cpu', adaptation=False)
    assert bundle.model is tiny
    assert bundle.tokenizer is tokenizer
    assert weights.call_args.kwargs['revision'] != 'main'
    assert weights.call_args.kwargs['config'] is config
    assert weights.call_args.kwargs['attn_implementation'] == 'eager'
