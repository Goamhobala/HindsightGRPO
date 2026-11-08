"""Pinned official MBPP data, one prompt, and conservative implementation spans."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

DATASET_ID = "google-research-datasets/mbpp"
DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
EXPECTED_SPLIT_SIZES = {"train": 374, "validation": 90, "test": 500, "prompt": 10}
PROMPT_TEMPLATE_VERSION = "mbpp-implementation-v1"
SYSTEM_PROMPT = "You are a Python programmer. Return only the requested Python implementation."


def load_mbpp(cache_dir: str | Path | None = None,
              manifest_path: str | Path | None = None) -> Any:
    """Load all official full splits without screening, then verify their boundaries."""
    from datasets import load_dataset

    data = load_dataset(DATASET_ID, "full", revision=DATASET_REVISION,
                        cache_dir=str(cache_dir) if cache_dir else None)
    actual = {name: len(split) for name, split in data.items()}
    if actual != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Official MBPP split-size mismatch: {actual}")
    id_sets = {name: set(split["task_id"]) for name, split in data.items()}
    for name, ids in id_sets.items():
        if len(ids) != len(data[name]):
            raise ValueError(f"Duplicate MBPP task IDs in {name}")
        for other in id_sets:
            if other != name and ids & id_sets[other]:
                raise ValueError(f"MBPP split leakage: {name}, {other}")
    manifest = {
        "dataset": DATASET_ID, "configuration": "full", "revision": DATASET_REVISION,
        "split_sizes": actual, "prompt_template": PROMPT_TEMPLATE_VERSION,
        "splits": {
            name: {"fingerprint": split._fingerprint,
                   "task_ids": list(map(int, split["task_id"])),
                   "content_sha256": hashlib.sha256(json.dumps(
                       list(split), sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
            for name, split in data.items()
        },
    }
    if manifest_path:
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(target)
    return data


def prompt_for(row: Mapping[str, Any], tokenizer: Any = None) -> str:
    """Use public task tests to name the function; never include the solution field."""
    task = str(row.get("text", row.get("prompt", ""))).strip()
    tests = "\n".join(row.get("test_list", []))
    user = (f"Write a Python implementation for this task:\n{task}\n\n"
            f"The implementation must satisfy these examples:\n{tests}\n\n"
            "Return Python code only, without tests or explanatory prose.")
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{user}\n\n"


def completion_text(token_ids: Sequence[int], tokenizer: Any) -> str:
    """Decode up to the first stop, while retaining all 256 tokens in history.

    Removing special strings after decoding would join post-stop samples to the
    implementation; leaving a terminal marker inline would create false syntax
    errors. Training and every evaluation phase use this same token boundary.
    """
    stop_ids = set()
    for field in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, field, None)
        if value is not None:
            stop_ids.update(value if isinstance(value, (list, tuple)) else [value])
    ids = [int(token) for token in token_ids]
    end = next((i for i, token in enumerate(ids) if token in stop_ids), len(ids))
    return tokenizer.decode(ids[:end], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(?P<code>.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)
_CODE_START = re.compile(r"^(?:async\s+def\s|def\s|class\s|import\s|from\s|@|[A-Za-z_]\w*\s*(?::[^=]+)?=)")
_TEST_LINE = re.compile(r"^(?:assert\s|if\s+__name__\s*==|(?:unittest|pytest)\.)")


def implementation_span(text: str) -> tuple[int, int]:
    """Return one contiguous implementation span, excluding identifiable test suffixes.

    Invalid or truncated Python remains rewardable (and can receive syntax=0).
    We never trim arbitrary broken syntax until it becomes a successful solution.
    """
    match = _FENCE.search(text)
    start, end = match.span("code") if match else (0, len(text))
    block = text[start:end]
    lines = block.splitlines(keepends=True)
    offset = 0
    first_code = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<|"):
            end = min(end, start + offset)
            break
        if first_code is None and _CODE_START.match(line):
            first_code = start + offset
        if first_code is not None and not line[:1].isspace() and _TEST_LINE.match(stripped):
            end = min(end, start + offset)
            break
        offset += len(line)
    if first_code is None:
        return start, start
    start = first_code
    # AST provides reliable top-level test boundaries for otherwise valid code.
    try:
        source = text[start:end]
        tree = ast.parse(source)
        line_starts = [0]
        for line in source.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))
        for node in tree.body:
            test_function = (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                and any(isinstance(child, ast.Assert) for child in ast.walk(node))
                and not any(isinstance(child, ast.Return) and child.value is not None
                            for child in ast.walk(node))
            )
            test_class = isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Attribute) and base.attr == "TestCase"
                or isinstance(base, ast.Name) and base.id == "TestCase"
                for base in node.bases
            )
            is_test = isinstance(node, ast.Assert) or test_function or test_class
            if is_test:
                end = start + line_starts[node.lineno - 1]
                break
    except (SyntaxError, ValueError):
        pass
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def extract_code(text: str) -> str:
    start, end = implementation_span(text)
    return text[start:end]


def implementation_token_mask(token_ids: Sequence[int], tokenizer: Any) -> list[bool]:
    """Map completed response tokens to code by exact tokenizer offsets.

    Boundaries require full token containment. Ambiguous decode/re-encode alignment
    fails closed, avoiding silently selecting non-code or a wrong token position.
    """
    ids = [int(v) for v in token_ids]
    result = [False] * len(ids)
    special = set(getattr(tokenizer, "all_special_ids", []))
    stop = {v for v in [getattr(tokenizer, "eos_token_id", None),
                        getattr(tokenizer, "pad_token_id", None)] if v is not None}
    end = next((i for i, v in enumerate(ids) if v in stop), len(ids))
    active = ids[:end]
    text = tokenizer.decode(active, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) != active:
            raise ValueError("decode/re-encode changed token boundaries")
        start_char, end_char = implementation_span(text)
        for i, (a, b) in enumerate(encoded["offset_mapping"]):
            result[i] = (ids[i] not in special and b > a and a >= start_char
                         and b <= end_char)
    except (TypeError, ValueError, NotImplementedError, KeyError) as error:
        warnings.warn(f"No eligible implementation tokens: {error}", RuntimeWarning)
    return result
