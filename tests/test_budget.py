import time
from types import SimpleNamespace

import pytest

from hindsight_grpo.budget import RunBudget, main
from scripts.wandb_status import snapshot


def test_test_subset_is_independent_of_arm_outputs_and_input_order():
    rows = [{'task_id': i} for i in range(500)]
    now = time.time()
    budget = RunBudget(200, 100, now, now + 36 * 3600, now + 48 * 3600)
    first = {r['task_id'] for r in budget.test_rows(rows, 20261108)}
    second = {r['task_id'] for r in budget.test_rows(list(reversed(rows)), 20261108)}
    assert first == second
    assert len(first) == 100
    assert {r['task_id'] for r in rows} == set(range(500))


@pytest.mark.parametrize('prompts', [0, 1, 199, 375, 400])
def test_cutoff_requires_complete_batches_within_first_epoch(prompts):
    with pytest.raises(ValueError, match='cutoff'):
        RunBudget(prompts, 100, 1, 2, 3)


def test_reactivation_preserves_original_deadlines(tmp_path, monkeypatch):
    args = ['budget', '--group', 'same-group', '--run-root', str(tmp_path)]
    monkeypatch.setattr('sys.argv', args)
    main()
    path = tmp_path / 'same-group/run_budget.json'
    before = path.read_bytes()
    monkeypatch.setattr('hindsight_grpo.budget.time.time', lambda: 99999999999)
    main()
    assert path.read_bytes() == before
    monkeypatch.setattr('sys.argv', args + ['--training-prompts', '196'])
    with pytest.raises(SystemExit):
        main()
    assert path.read_bytes() == before


def test_wandb_export_queries_only_group_and_keeps_progress_without_configuration():
    calls = []
    run = SimpleNamespace(name='root', id='run-id', state='running', url='https://example.invalid',
        summary={'latest_update': 32, 'progress/phase': 'training',
                 'checkpoint/prompts_processed': 131, 'unrelated_private_data': 'exclude'})
    def runs(project, filters):
        calls.append((project, filters))
        return [run]
    rows = snapshot(SimpleNamespace(runs=runs), 'entity', 'project', 'group')
    assert calls == [('entity/project', {'group': 'group'})]
    assert rows[0]['summary'] == {'latest_update': 32, 'progress/phase': 'training',
                                'checkpoint/prompts_processed': 131}
