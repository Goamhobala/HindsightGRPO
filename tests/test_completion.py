from types import SimpleNamespace

from hindsight_grpo.data import completion_text


def test_stop_marker_and_post_stop_samples_never_enter_reward_code():
    tokens = [1, 2, 9, 3, 4]
    tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=8,
        decode=lambda ids, **kwargs: ''.join({1:'def f():\n', 2:'    return 1',
            9:'<|im_end|>', 3:'garbage', 4:'<|endoftext|>'}[i] for i in ids))
    assert completion_text(tokens, tokenizer) == 'def f():\n    return 1'
    assert tokens == [1, 2, 9, 3, 4]  # The complete trajectory is retained.


def test_multiple_stop_ids_and_no_stop_token():
    tokenizer = SimpleNamespace(eos_token_id=[8, 9], pad_token_id=7,
                                decode=lambda ids, **kwargs: str(ids))
    assert completion_text([1, 8, 2], tokenizer) == '[1]'
    assert completion_text([1, 7, 2], tokenizer) == '[1]'
    assert completion_text([1, 2], tokenizer) == '[1, 2]'
